from qwen import model as qwenjax
from text._sampler import SamplingState, _slice_cache, _merge_cache, _get_last_token_pos, _get_last_token, _make_full_attention_mask
from . import _sampling

import functools
import dataclasses
from functools import partial

import einops
import jax
import jax.numpy as jnp
from jax import random

def generate(
    model: qwenjax.Weights,
    tokens: jax.Array,
    position_ids: jax.Array,
    attention_mask: jax.Array,
    cfg: qwenjax.Config,
    sampling: _sampling.CombinedSampling,
    rng: jax.random.PRNGKey,
    max_new_tokens: int,
    cache_length: int,
):
    """Generate multiple answers for a given question."""
    batch_size, _ = tokens.shape
    init_cache_length = tokens.shape[-1]
    if init_cache_length > cache_length:
        raise ValueError(
            'Cache buffer filled up. With the new input, it uses:'
            f' {init_cache_length}/{cache_length} tokens.'
        )
    remaining_cache_length = cache_length - init_cache_length + 1
    max_new_tokens = min(max_new_tokens, remaining_cache_length)
    cache = qwenjax.KVCache.init(
        random.key(1),
        cfg,
        batch_size=len(tokens),
        cache_size=cache_length,
    )
    logits, new_cache = qwenjax.forward(
        x=tokens,
        segment_pos=position_ids,
        attn_mask=attention_mask,
        weights=model,
        cfg=cfg,
        cache=_slice_cache(cache, length=init_cache_length),
    )
    cache = _merge_cache(
        old_cache=cache,
        new_cache=new_cache,
        length=init_cache_length,
    )
    last_token_pos = _get_last_token_pos(tokens)
    last_token = _get_last_token(tokens)
    
    full_attention_mask = _make_full_attention_mask(
        tokens=tokens,
        cache_length=cache_length,
    )
    state = SamplingState(
        step=jnp.asarray(0),
        done=jnp.zeros((batch_size,), dtype=jnp.bool_),
        last_token=last_token,
        last_token_pos=last_token_pos,
        predicted_tokens=jnp.zeros(
            (batch_size, max_new_tokens), dtype=jnp.int32
        ),
        cache=cache,
        rng=jax.random.split(rng, tokens.shape[0]),
        full_attention_mask=full_attention_mask,
        init_cache_length=jnp.asarray(init_cache_length),
    )   
    state = _sample_loop(
        state=state,
        weights=model,
        sampling=sampling,
        max_new_tokens=max_new_tokens,
        cache_length=cache_length,
        cfg=cfg,
    ) 
    return {
        'completion_ids': state.predicted_tokens,
        'completion_mask': _mask_tokens_after_end_tokens(
            state.predicted_tokens,
            cfg.eos_token_id,
        )
    }
    

def _sample_loop(
    state: SamplingState,
    weights: qwenjax.Weights,
    sampling: _sampling.CombinedSampling,
    max_new_tokens: int,
    cache_length: int,
    cfg: qwenjax.Config,
) -> SamplingState:
    step_fn = functools.partial(_sample_step, 
                                weights=weights, sampling=sampling, cache_length=cache_length, cfg=cfg)
    def cond_fn(state: SamplingState):
        return (state.step < max_new_tokens) & ~jnp.all(state.done)
    
    state = jax.lax.while_loop(cond_fn, step_fn, state)
    return state

@partial(jax.jit, static_argnums=(2,3)) 
def _sample_step(
    state: SamplingState,
    weights: qwenjax.Weights,
    sampling: _sampling.CombinedSampling,
    cache_length: int,
    cfg: qwenjax.Config,
) -> SamplingState:
    step_mask = jnp.arange(cache_length) < state.used_cache_length
    attention_mask = state.full_attention_mask * step_mask
    attention_mask = einops.rearrange(attention_mask, 'B L -> B 1 L')
    logits, new_cache = qwenjax.forward(
        x=state.last_token[..., None],
        segment_pos=state.last_token_pos[..., None],
        attn_mask=attention_mask,
        weights=weights,
        cfg=cfg,
        cache=state.cache,
    )
    logits = einops.rearrange(logits, 'B 1 V -> B V')
    
    next_rngs, curr_rngs = _split_batch_rngs(state.rng)
    next_tokens = jax.vmap(sampling.sample)(logits=logits, rng=curr_rngs)
    
    predicted_tokens = state.predicted_tokens.at[:, state.step].set(next_tokens)
    done = state.done | jnp.isin(next_tokens, jnp.asarray(cfg.eos_token_id))
    
    return SamplingState(
        step=state.step+1,
        done=done,
        last_token=next_tokens,
        last_token_pos=state.last_token_pos + ~state.done,
        predicted_tokens=predicted_tokens,
        cache=new_cache,
        rng=next_rngs,
        init_cache_length=state.init_cache_length,
        full_attention_mask=state.full_attention_mask,
    )
    
def _split_batch_rngs(rngs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split a batch of RNGs into two new batches."""
    split = jax.vmap(jax.random.split)(rngs)  # (B, 2, 2)
    return split[:, 0], split[:, 1]

def _mask_tokens_after_end_tokens(
    tokens,
    end_tokens,
):
    """Mask token IDs after the first EOS token with 0."""
    end_tokens_mask = jnp.isin(tokens, jnp.asarray(end_tokens))
    end_tokens_mask = jnp.cumsum(end_tokens_mask, axis=-1) - end_tokens_mask == 0
    return end_tokens_mask