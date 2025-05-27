import functools
from collections.abc import Iterator, Sequence
import dataclasses

import jax
from transformers import PreTrainedTokenizer

from qwen import model as qwenjax
from text import _sampler
from text import _sampling, _sampler_call, _template

@dataclasses.dataclass(frozen=True, kw_only=True, eq=False)
class ChatSampler:
    model: qwenjax.Weights
    cfg: qwenjax.Config
    tokenizer: PreTrainedTokenizer
    sampling: _sampling.CombinedSampling 
    multi_turn: bool = False
    print_stream: bool = False
    cache_length: int | None = 4096
    max_out_length: int = 2048
    last_state: _sampler_call.SamplingState = dataclasses.field(default=None, 
                                                           repr=False
    )
    turns: list[_template.Turn] = dataclasses.field(default_factory=list)
    
    def __post_init__(self):
        if self.turns:
            raise ValueError(
            'Currently initializing the sampler with previous conversation is not'
            ' supported.'
            )
        object.__setattr__(self, 'last_state', None)
    
    @functools.cached_property
    def sampler(self) -> _sampler.Sampler:
        return _sampler.Sampler(
            model=self.model,
            cfg=self.cfg,
            tokenizer=self.tokenizer,
            sampling=self.sampling,
            cache_length=self.cache_length,
            max_out_length=self.max_out_length,
        )
    
    def chat(
        self,
        prompt: str,
        sampling: _sampling.CombinedSampling,
        *,
        rng: jax.Array | None = None,
        max_new_tokens: int | None = None,
        multi_turn: bool | None = None,
        print_stream: bool | None = None,
    ):
        if multi_turn is None:
            multi_turn = self.multi_turn
        if print_stream is None:
            print_stream = self.print_stream
            
        unformatted_prompt = prompt
        
        prompt = [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        
        if not multi_turn:
            object.__setattr__(self, 'last_state', None)
            object.__setattr__(self, 'turns', [])    
            
        out = self.sampler.sample(
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            sampling=sampling,
            rng=rng,
            return_state=True,
            stream=print_stream
        )

        if print_stream:
            out = _print_stream(out)
        assert isinstance(out, _sampler.SamplerOutput) 
        assert isinstance(out.text, str) 
        model_output = out.text.removesuffix('<|im_end|>')
        
        self.turns.append(_template.UserTurn(unformatted_prompt))
        self.turns.append(_template.ModelTurn(model_output))
        object.__setattr__(self, 'last_state', out.state)
        return model_output
            

def _print_stream(
    out: Iterator[_sampler.SamplerOutput],
) -> _sampler.SamplerOutput:
    """Prints the streaming output."""
    text_tokens = []
    for state in out:
        text_tokens.append(state.text)
        if state.text == '<|im_end|>':  # Last token is not printed.
            continue
        print(state.text, end='', flush=True)
    out = dataclasses.replace(state, text=''.join(text_tokens))  
    return out
            