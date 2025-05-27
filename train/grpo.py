from .trainer import generate
from ._sampling import CombinedSampling
from qwen import model as qwenjax
from text._sampler import  _encode_prompts, build_positions_from_mask, _make_causal_mask
from . import evaluator 
from .utils import _get_per_token_logps

from typing import Dict, Any
import jax
import jax.numpy as jnp
from transformers import PreTrainedTokenizer
from dataclasses import dataclass, field
import itertools

@dataclass(kw_only=True)
class GRPO:
    model: qwenjax.Weights
    cfg: qwenjax.Config
    tokenizer: PreTrainedTokenizer
    cache_length: int | None
    evaluator: evaluator.RewardEvaluator
    rng: jax.random.PRNGKey
    temperature: float
    top_p: float | None
    top_k: int | None
    min_p: float | None
    num_generations: int
    max_prompt_length: int
    max_new_tokens: int
    num_iterations: int
    epsilon: float
    beta: float
    use_clipping: bool = False
    ref_model: qwenjax.Weights | None = field(init=False, default=None)
    sampling: CombinedSampling | None = field(init=False, default=None)
    
    def __post_init__(self):
        if self.beta == 0.0:
            self.ref_model = None
        else:
            self.ref_model = qwenjax.create_reference_model(self.model)
            
        self.sampling = CombinedSampling(
            top_k=self.top_k,
            top_p=self.top_p,
            min_p=self.min_p,
            temperature=self.temperature
        )
    
    def _prepare_inputs(
        self,
        prompts: list[list[Dict[str, str]]],
        answers: list[str],
    ):
        inputs, metrics, log_data = self._generate_and_score_completions(prompts, answers)
        return inputs, metrics, log_data
    
    def _generate_and_score_completions(
        self,
        prompts: list[list[Dict[str, str]]],
        answers: list[str],
    ):
        prompts_text = [self.tokenizer.apply_chat_template(prompt, tokenize=False) for prompt in prompts]
        prompt_ids, _ = _encode_prompts(prompts_text, self.tokenizer, self.max_prompt_length)
        prompt_ids = jnp.repeat(prompt_ids, self.num_generations, axis=0)
        prompt_mask = prompt_ids != self.tokenizer.pad_token_id
        position_ids = build_positions_from_mask(prompt_mask)
        attention_mask = _make_causal_mask(prompt_mask)
        output = generate(
            model=self.model,
            tokens=prompt_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            cfg=self.cfg,
            sampling=self.sampling,
            rng=self.rng,
            max_new_tokens=self.max_new_tokens,
            cache_length=self.cache_length
        )
        completion_ids = output['completion_ids']
        completion_mask = output['completion_mask']
        prompt_completion_mask = jnp.concat([prompt_mask, output["completion_mask"]], axis=1)
        attention_mask = _make_causal_mask(prompt_completion_mask)
        prompt_completion_ids = jnp.concat([prompt_ids, output['completion_ids']], axis=1)
        position_ids = build_positions_from_mask(prompt_completion_mask)
        logits_to_keep = completion_ids.shape[1]
        if self.num_iterations > 1:
            old_per_token_logps = _get_per_token_logps(
                model=self.model, 
                input_ids=prompt_completion_ids, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                cfg=self.cfg, 
                logits_to_keep=logits_to_keep)
        else:
            old_per_token_logps = None
        
        ref_per_token_logps = None
        if self.beta != 0.0:
            if self.ref_model is not None:
                ref_per_token_logps = _get_per_token_logps(
                    model=self.ref_model,
                    input_ids=prompt_completion_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    cfg=self.cfg,
                    logits_to_keep=logits_to_keep,
                )        
        
        completions_text = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        advantages, metrics, log_data = self._score_completions(completions_text, prompts_text, answers)
        return {
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'completion_ids': completion_ids,
            'completion_mask': completion_mask,
            'prompt_completion_ids': prompt_completion_ids,
            'prompt_completion_mask': prompt_completion_mask,
            'advantages': advantages,
            'old_per_token_logps': old_per_token_logps,
            'ref_per_token_logps': ref_per_token_logps,
        }, metrics, log_data
    
    def _score_completions(
        self,
        completions_text: list[str],
        questions: list[str],
        answers: list[str],
    ):
        log_data = [
            {
                'prompt': {
                    'text': question,
                    'answer': answer
                },
                'generations': []
            }
            for question, answer in zip(questions, answers)
        ]
        mock_completions = [[{'content': completion}] for completion in completions_text]
        answers = list(itertools.chain.from_iterable(itertools.repeat(ans, self.num_generations) for ans in answers))
        rewards_per_func, metrics = self.evaluator.compute_rewards(
            completions=mock_completions,
            answers=answers,
        )
        rewards = rewards_per_func.sum(axis=1)
        
            # Store generation data
        for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
            generation_data = {
                'response': completion,
                'scores': {
                   **self.evaluator.get_reward_breakdown(reward_scores),
                    'total_reward': rewards[i].item()
                }
            }
            # Determine the index of the original question
            q_idx = i // self.num_generations
            log_data[q_idx]['generations'].append(generation_data)
        
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.reshape(-1, self.num_generations).mean(axis=1)
        std_grouped_rewards = rewards.reshape(-1, self.num_generations).std(axis=1)
        
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = jnp.repeat(mean_grouped_rewards, self.num_generations, axis=0)
        std_grouped_rewards = jnp.repeat(std_grouped_rewards, self.num_generations, axis=0)
        
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        metrics["reward_std"] = std_grouped_rewards.mean().item() 
        
        return advantages, metrics, log_data
    
def compute_loss(model, inputs, cfg, use_clipping, beta, epsilon):
    prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
    completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
    prompt_completion_ids, prompt_completion_mask = inputs["prompt_completion_ids"], inputs["prompt_completion_mask"]
    
    logits_to_keep = completion_ids.shape[1] 
    attention_mask = _make_causal_mask(prompt_completion_mask)
    position_ids = build_positions_from_mask(prompt_completion_mask)
    per_token_logps = _get_per_token_logps(
        model=model,
        input_ids=prompt_completion_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cfg=cfg,
        logits_to_keep=logits_to_keep,
    )
    if inputs["ref_per_token_logps"] is not None:
        per_token_kl = jnp.exp(inputs["ref_per_token_logps"] - per_token_logps) - (inputs["ref_per_token_logps"] - per_token_logps) - 1
    else:
        ref_per_token_logps = _get_per_token_logps(model, prompt_completion_ids, attention_mask, position_ids, cfg, logits_to_keep)
        per_token_kl = jnp.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1            

    advantages = inputs["advantages"]
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps) if inputs["old_per_token_logps"] is None else inputs["old_per_token_logps"]

    probability_ratio = jnp.exp(per_token_logps - old_per_token_logps)
    if use_clipping:
        per_token_loss = -jnp.min(
            probability_ratio * advantages[:, None],
            jnp.clip(probability_ratio, 1 - epsilon, 1 + epsilon) * advantages[:, None])
    else:
        per_token_loss = probability_ratio * advantages[:, None]
    if beta != 0.0:
        per_token_loss = per_token_loss + beta * per_token_kl
    
    loss = ((per_token_loss * completion_mask).sum(axis=-1) / completion_mask.sum(axis=-1)).mean()
    
    # Additional_metrics
    metrics = {}
    response_length = completion_mask.sum(axis=-1).mean()
    metrics["response_length"] = response_length
    mean_kl = ((per_token_kl * completion_mask).sum(axis=-1) / completion_mask.sum(axis=-1)).mean()
    metrics["mean_kl"] = mean_kl
    
    return loss, metrics

    