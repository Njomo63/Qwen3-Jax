import qwen.model as qwenjax
from . import _sampling
from . import _sampler_call

from typing import Iterator
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from typing import Sequence
import dataclasses
from transformers import PreTrainedTokenizer
import random as py_random
   

@dataclasses.dataclass(frozen=True, kw_only=True)
class SamplerOutput:
    """Output of the sampler when `return_state=True`.

    Attributes:
        text: Sampled text.
        state: State for extra information, and which can be used in the next turn.
    """
    text: str | list[str]
    state: _sampler_call.SamplingState
    
    @property
    def tokens(self):
        """Predicted tokens."""
        return self._maybe_unbatch(self.state.predicted_tokens)
    
    def _maybe_unbatch(self, x):
        if isinstance(self.text, str):
            (x,) = x
        return x

@dataclasses.dataclass(frozen=True, kw_only=True)
class Sampler:
    model: qwenjax.Weights
    cfg: qwenjax.Config
    sampling: _sampling.CombinedSampling
    tokenizer: PreTrainedTokenizer
    cache_length: int = 4096
    max_out_length: int = 2048
    pad_length: int | None = None
    
    def sample(self,
               prompt: str | Sequence[str],
               *,
               max_new_tokens: int | None = None,
               stream: bool = False,
               sampling = None,
               rng = None,
               return_state: bool = False,
               last_state: _sampler_call.SamplingState | None = None,
    ):
        sampling = sampling or self.sampling
        rng = _normalize_rng(rng)
        tokens, is_single_prompt = self._encode_prompts(
            prompt=prompt,
            pad_length=self.pad_length,
        )
        init_cache_length = tokens.shape[-1]
        if last_state is not None:
            init_cache_length += int(last_state.used_cache_length)
        if init_cache_length > self.cache_length:
            raise ValueError(
                'Cache buffer filled up. With the new input, it uses:'
                f' {init_cache_length}/{self.cache_length} tokens.'
            )
        remaining_cache_length = self.cache_length - init_cache_length + 1
        max_new_tokens = max_new_tokens or self.max_out_length
        max_new_tokens = min(max_new_tokens, remaining_cache_length)            
        if last_state is None:
            cache = qwenjax.KVCache.init(
                random.key(1),
                self.cfg,
                batch_size = len(tokens),
                cache_size = self.cache_length,
            )
        else:
            cache = last_state.cache
            
        sampler = _sampler_call.SamplerCall(
            end_tokens=(
                self.cfg.eos_token_id,
                self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            ),
            padding_id=self.tokenizer.pad_token_id,
            sampling=sampling,
            cache_length=self.cache_length,
            max_out_length=self.max_out_length
        )
        
        state = sampler.sample(
            model=self.model,
            cfg=self.cfg,
            tokens=tokens,
            cache=cache,
            last_state=last_state,
            max_new_tokens=jnp.asarray(max_new_tokens),
            init_cache_length=init_cache_length,
            rng=rng,
            stream=stream
        )
        
        if stream:
            return self._stream_decode_state(
                state,
                return_state=return_state
        )
        else:
            return self._decode_state(
                state,
                predicted_tokens=state.predicted_tokens,
                is_single_prompt=is_single_prompt,
                return_state=return_state
        )
    
    def _encode_prompts(
        self,
        prompt: str | Sequence[str],
        pad_length: int | None = None,
    ) -> tuple[jax.Array, bool]:
        """Encode the prompt"""
        prompt, is_single_prompt = _normalize_prompt(prompt)
        tokens = [self.tokenizer(p)["input_ids"] for p in prompt]
        
        max_prompt_len = pad_length or max(len(t) for t in tokens)
        
        tokens = pad(tokens, max_length=max_prompt_len, fill_value=self.tokenizer.pad_token_id)
        tokens = jnp.asarray(tokens)
        return tokens, is_single_prompt
    
    def _decode_state(
        self,
        state: _sampler_call.SamplingState,
        predicted_tokens: jax.Array,
        is_single_prompt: bool,
        return_state: bool,
    ):
        predicted_texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in predicted_tokens.tolist()]
        if is_single_prompt:
            (predicted_texts,) = predicted_texts
        
        if return_state:
            return SamplerOutput(
                text=predicted_texts,
                state=state,
            )
        else:
            return predicted_texts

    def _stream_decode_state(
        self,
        state_iter: Iterator[_sampler_call.SamplingState],
        return_state: bool,
    ):
        for i, state in enumerate(state_iter):
            yield self._decode_state(
                state,
                predicted_tokens=state.predicted_tokens[..., i],
                is_single_prompt=True,
                return_state=return_state
            )


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

def encode(
    text: str | list[str],
    tokenizer: PreTrainedTokenizer,
) -> list[int]:
    """Encode  a text into a list of token ids.
    
    Args:
        text: The text to encode. Can be a string or a list of strings.
    Returns:
        A list of token ids.
    """
    if isinstance(text, str):
        token_ids = tokenizer(text)["input_ids"]
    else:
        token_ids = tokenizer(
            text, 
            is_split_into_words=True)["input_ids"]
    return token_ids

    

def _normalize_prompt(prompt: str | Sequence[str]) -> tuple[list[str], bool]:
    """Normalize the inputs."""
    if _is_str_array(prompt):  # Supports batched input array
        assert isinstance(prompt, np.ndarray)
        prompt = prompt.tolist()

    if isinstance(prompt, str):
        is_single_prompt = True
        prompt = [prompt]
    else:
        is_single_prompt = False
        prompt = list(prompt)

    return prompt, is_single_prompt

def _normalize_rng(seed_or_rng: jax.Array | None) -> jax.random.PRNGKey:
    if seed_or_rng is None:
        seed_or_rng = py_random.randint(0, 1000000000)
    if not isinstance(seed_or_rng, jax.Array):
        seed_or_rng = jax.random.key(seed_or_rng)
    return seed_or_rng

def _is_str_array(x) -> bool:
    if not isinstance(x, np.ndarray):
        return False
    return np.dtype(x.dtype).type in {np.object_, np.str_}
