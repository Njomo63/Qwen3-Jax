from etils import epath
from pprint import pprint
import argparse

import jax
import jax.numpy as jnp

from qwen import model as qwenjax
from text import _chat_sampler, _sampling

def parse_args():
    parser = argparse.ArgumentParser(description="Decoding Script")
    parser.add_argument("--model_path", type=str, default="/model", help="Path of model")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling parameter")
    parser.add_argument("--min_p", type=float, default=0.05, help="Min-p sampling parameter")  
    
    args = parser.parse_args()
    return args  
    

if __name__ == "__main__":
    args = parse_args()
    ckpt_path = epath.Path(args.model_path)
    
    tokenizer = qwenjax.load_tokenizer(ckpt_path)
    
    cfg = qwenjax.load_config(ckpt_path / "config.json")
    model = qwenjax.load_pytree(ckpt_path, qwenjax.Weights.init_placeholder(cfg))
    sampling = _sampling.CombinedSampling(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p
    )
    sampler = _chat_sampler.ChatSampler(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        sampling=sampling,
        print_stream=True,
    )
    prompt = "What is the capital city of France?"
    rng = jax.random.PRNGKey(0)
    output = sampler.chat(
        prompt=prompt,
        sampling=sampling,
        rng=rng,
        max_new_tokens=50,
        print_stream=True,
    )