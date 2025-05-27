import qwen.model as qwenjax
from . import _sampling

from typing import Iterator
import functools
from functools import partial
import dataclasses

import einops
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Bool, Int

@qwenjax.jax_pytree_struct
class SamplingState:
    """Internal sampling state.

    Attributes:
        step: Number of steps decoding steps taken so far (between [0,
        max_new_tokens]).
        done: For each sequence in the batch, `True` if the sequence is done (i.e
        has predicted a `<eos>` token).
        last_token: Model input for the next sampling step.
        last_token_pos: The RoPE position of the last token in the input.
        predicted_tokens: Fixed-size buffer for accumulating the output tokens.
        full_attention_mask: Pre-computed attention mask for the full sequence.
        cache: Attention KV cache.
        rng: Seed to use for sampling.
        init_cache_length: Length of the cache length in the pre-fill phase. Include
        the prompt and the previous turns.
    """
    step: Int[Array, ""]
    done: Bool[Array, "B"]
    last_token: Int[Array, "B"]
    last_token_pos: Int[Array, "B"]
    predicted_tokens: Int[Array, "B max_out_length"]
    full_attention_mask: Bool[Array, "B cache_length"]
    cache: qwenjax.KVCache
    rng: jax.random.PRNGKey
    init_cache_length: Int[Array, ""]
    
    @property
    def used_cache_length(self):
        """Length of the cache currently used."""
        return self.init_cache_length + self.step
    
    
@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplerCall:
    end_tokens: tuple[int, ...]
    padding_id: int
    sampling: _sampling.CombinedSampling
    cache_length: int
    max_out_length: int
    

    def sample(
        self,
        model: qwenjax.Weights,
        cfg: qwenjax.Config,
        tokens: Int[Array, "B L"],
        cache: qwenjax.KVCache,
        last_state: SamplingState | None,
        max_new_tokens: Int[Array, ""],
        init_cache_length: int,
        rng: jax.random.PRNGKey,
        stream: bool = False,
    ) -> SamplingState | Iterator[SamplingState]:
        
        # Prefill the KV cache with the prompt
        init_state = self._prefill(
            model=model,
            cfg=cfg,
            tokens=tokens,
            init_cache_length=init_cache_length,
            cache=cache,
            last_state=last_state,
            rng=rng
        )
        
        sample_fn = self._stream_sample_loop if stream else self._sample_loop
        
        state = sample_fn(
            model=model,
            cfg=cfg,
            state=init_state,
            max_new_tokens=max_new_tokens,
        )
        
        return state
    
    def _prefill(
        self,
        *,
        model: qwenjax.Weights,
        cfg: qwenjax.Config,
        tokens: Int[Array, "B L"],
        init_cache_length: int,
        cache: qwenjax.KVCache,
        last_state: SamplingState | None,
        rng: jax.random.PRNGKey
    ) -> SamplingState:
        """Prefills the KVCache"""
        batch_size, _ = tokens.shape
        inputs_mask = tokens != self.padding_id
        
        if last_state is None:
            positions_offset = None
            attention_mask = None
        else:
            positions_offset = last_state.last_token_pos
            attention_mask = _make_multi_turn_attention_mask(
                last_state=last_state,
                inputs_mask=inputs_mask,
            )
            
        positions = build_positions_from_mask(inputs_mask)
        if positions_offset is not None:
            positions += positions_offset[..., None]
        if attention_mask is None:
            attention_mask = _make_causal_mask(inputs_mask)
        logits, new_cache = jax.jit(qwenjax.forward, donate_argnums=5)(
            x=tokens,
            segment_pos=positions,
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
        last_token_pos = _get_last_token_pos(inputs_mask)
        if positions_offset:
            last_token_pos += positions_offset
        last_token = _get_last_token(tokens, inputs_mask)
        
        full_attention_mask = _make_full_attention_mask(
            cache_length=self.cache_length,
            inputs_mask=inputs_mask
        )
        return SamplingState(
            step=jnp.asarray(0),
            done=jnp.zeros((batch_size,), dtype=jnp.bool_),
            last_token=last_token,
            last_token_pos=last_token_pos,
            predicted_tokens=jnp.zeros(
                (batch_size, self.max_out_length), dtype=jnp.int32
            ),
            cache=cache,
            rng=rng,
            full_attention_mask=full_attention_mask,
            init_cache_length=jnp.asarray(init_cache_length),
        )
        
    @partial(jax.jit, static_argnames=('self')) 
    def _sample_step(
        self,
        *,
        state: SamplingState,
        model: qwenjax.Weights,
        cfg: qwenjax.Config,
    ) -> SamplingState:
        step_mask = jnp.arange(self.cache_length) < state.used_cache_length
        attention_mask = state.full_attention_mask * step_mask
        attention_mask = einops.rearrange(attention_mask, 'B L -> B 1 L')
        logits, new_cache = qwenjax.forward(
            x=state.last_token[..., None],
            segment_pos=state.last_token_pos[..., None],
            attn_mask=attention_mask,
            weights=model,
            cfg=cfg,
            cache=state.cache,
        )
        logits = einops.rearrange(logits, 'B 1 V -> B V')
        
        next_rng, curr_rng = jax.random.split(state.rng)
        next_token = self.sampling.sample(logits=logits, rng=curr_rng)
        
        predicted_tokens = state.predicted_tokens.at[:, state.step].set(next_token)
        done = state.done | jnp.isin(next_token, jnp.asarray(self.end_tokens))
        
        return SamplingState(
            step=state.step+1,
            done=done,
            last_token=next_token,
            last_token_pos=state.last_token_pos + ~state.done,
            predicted_tokens=predicted_tokens,
            cache=new_cache,
            rng=next_rng,
            init_cache_length=state.init_cache_length,
            full_attention_mask=state.full_attention_mask,
        )
        
    def _stream_sample_loop(
        self,
        *,
        state: SamplingState,
        model: qwenjax.Weights,
        cfg: qwenjax.Config,
        max_new_tokens: Int[Array, ""],
    )-> Iterator[SamplingState]:
        for _ in range(max_new_tokens):
            state = self._sample_step(
                state=state,
                model=model,
                cfg=cfg,
            )
            yield state
            if state.done[0].tolist():
                break
            
    def _sample_loop(
        self,
        *,
        state: SamplingState,
        model: qwenjax.Weights,
        cfg: qwenjax.Config,
        max_new_tokens: Int[Array, ""],
    ) -> SamplingState:
        step_fn = functools.partial(self._sample_step, 
                                    model=model, cfg=cfg)
        def cond_fn(state: SamplingState):
            return (state.step < max_new_tokens) & ~jnp.all(state.done)
        
        state = jax.lax.while_loop(cond_fn, step_fn, state)
        predicted_tokens = _mask_tokens_after_end_tokens(
            state.predicted_tokens, 
            self.end_tokens,
            self.padding_id
        )
        state = dataclasses.replace(
            state,
            predicted_tokens=predicted_tokens,
        )
        return state
        

def build_positions_from_mask(input_mask: Bool[Array, "B L"]) -> Int[Array, "B L"]:
    """Computes the `positions` from the `input_mask`.

    Args:
        input_mask: The tokens `input_mask`, True for non-padded tokens only.

    Returns:
        The indices to use for RoPE and absolute position encodings for the given
        input mask.
    """
    positions = jnp.cumsum(input_mask, axis=-1)
    # Subtract one for all positions from the first valid one as they are
    # 0-indexed
    return positions - (positions >= 1)

def _make_multi_turn_attention_mask(
    last_state: SamplingState,
    inputs_mask: Bool[Array, "B L"],
) -> Bool[Array, "B L L+used_cache_length"]:
    """Make the attention mask for the next prompt."""
    next_prompt_attn_mask = _make_causal_mask(inputs_mask)
    
    cache_att_mask = _make_cache_mask(
        state = last_state,
        next_prompt_length = next_prompt_attn_mask.shape[-1],
    )
    
    return jnp.concat([cache_att_mask, next_prompt_attn_mask], axis=-1)

def _make_cache_mask(
    state: SamplingState,
    next_prompt_length: int,
) -> Bool[Array, "B L used_cache_length"]:
    """Slice the previous attention mask from the KV cache."""
    cache_att_mask = state.full_attention_mask[:, :state.used_cache_length]
    cache_att_mask = cache_att_mask[:, None, :]  # b 1 used_cache_length
    cache_att_mask = jnp.broadcast_to(
        cache_att_mask,
        (
            cache_att_mask.shape[0],  # b
            next_prompt_length,  # L
            cache_att_mask.shape[2],  # used_cache_length
            ),
        )
    return cache_att_mask

def _make_causal_mask(
    input_mask: Bool[Array, "B L"],
) -> Bool[Array, "B L L"]:
    """Makes a causal attention mask.


    Args:
        input_mask: Input mask for the input. True for non-padded tokens only, else
        False.

    Returns:
        Attention mask of shape [B, L, L] (where B=batch dim and L=sequence dim).
    """
    if len(input_mask.shape) != 2:
        raise ValueError(
        f'Input mask must be 2D (shape [B, L]), but got {input_mask.shape}.'
    )
    seq_len = input_mask.shape[-1]
    causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool))
    attn_mask = input_mask[..., None, :]
    attn_mask *= causal_mask[None, ...]
    return attn_mask

def _make_full_attention_mask(
    cache_length: int,
    inputs_mask: Bool[Array, "B L"],
) -> Bool[Array, "B cache_length"]:
    
    # Pad the mask to the full `cache_length` for static shape.
    full_attention_mask = pad(
        inputs_mask,
        max_length=cache_length,
        fill_value=True
    )
    return full_attention_mask

def _slice_cache(cache: qwenjax.KVCache, length: int) -> qwenjax.KVCache:
    new_k_cache = []
    new_v_cache = []
    
    for k_cache, v_cache in zip(cache.k_cache, cache.v_cache):
        new_k_cache.append(k_cache[:, :length, :, :])
        new_v_cache.append(v_cache[:, :length, :, :])
    
    return dataclasses.replace(
        cache,
        k_cache=new_k_cache,
        v_cache=new_v_cache,
    )

def _merge_cache(
    old_cache: qwenjax.KVCache,
    new_cache: qwenjax.KVCache,
    length: int,
) -> qwenjax.KVCache:
    updated_k_cache = []
    updated_v_cache = []
    updated_end_index = []
    
    for idx, (old_k, old_v, new_k, new_v) in enumerate(zip(
        old_cache.k_cache, old_cache.v_cache, new_cache.k_cache, new_cache.v_cache
    )):
        updated_end_index.append(new_cache.end_index[idx] - 1)
        updated_k_cache.append(old_k.at[:, :length, :, :].set(new_k))
        updated_v_cache.append(old_v.at[:, :length, :, :].set(new_v))
    
    return dataclasses.replace(
        old_cache,
        k_cache = updated_k_cache,
        v_cache = updated_v_cache,
        end_index = updated_end_index,
    )
        
def _get_last_token_pos(
    inputs_mask: Bool[Array, "B L"] 
    ) -> Int[Array, "B"]:
    return jnp.sum(inputs_mask.astype(jnp.int32), axis=-1) - 1

def _get_last_token(
    tokens: Float[Array, "B L"],
    inputs_mask: Bool[Array, "B L"]) -> Int[Array, "B"]:
    last_token_pos = _get_last_token_pos(inputs_mask)
    x = jnp.take_along_axis(tokens, last_token_pos[:, None], axis=-1)
    x = jnp.squeeze(x, axis=-1)
    return x

def _mask_tokens_after_end_tokens(
    tokens: Int[Array, "B L"],
    end_tokens: tuple[int, ...],
    padding_id: int
):
    """Mask token IDs after the EOS token with the padding ID."""
    end_tokens_mask = jnp.isin(tokens, jnp.asarray(end_tokens))
    end_tokens_mask = jnp.cumsum(end_tokens_mask, axis=-1) - end_tokens_mask == 0
    return jnp.where(end_tokens_mask, tokens, padding_id)    

def _is_list_array(x) -> bool:
    """Returns `True` if `x` is a list of ints, like `[0, 1, ...]`."""
    return isinstance(x, list | tuple) and all(isinstance(x, int) for x in x) 

def pad(
    element: list,
    max_length: int,
    fill_value: int,
    truncate: bool = False,
    axis: int = -1,  
):
    if axis != -1:
        raise NotImplementedError("Only `axis=-1` is supported.")
    return jax.tree.map(
        lambda x: _pad(
            x,
            max_length=max_length,
            fill_value=fill_value,
            truncate=truncate,
        ),
        element,
        is_leaf=_is_list_array,
    )
def _pad(
    element: list,
    max_length: int,
    fill_value: int,
    truncate: bool = False,
):
    element = jnp.asarray(element)
    seq_len = element.shape[-1]
    if not truncate and seq_len > max_length:
        raise ValueError(
            f"Cannot pad sequence of length {seq_len}. Is longer than the"
            f" max length {max_length}. Set `truncate=True`."
    )
    sentence_tokens = element[..., :max_length]
    pad_width = [(0, 0)] * (sentence_tokens.ndim - 1) + [
        (0, max_length - sentence_tokens.shape[-1])
    ]
    return jnp.pad(sentence_tokens, pad_width, constant_values=fill_value)  