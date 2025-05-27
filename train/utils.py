from typing import Dict, Any, List

import qwen.model as qwenjax 
from jaxtyping import Float, Int, Array
import jax
import jax.numpy as jnp

def write_generation_log(log_data: List[Dict[str, Any]], log_file: str) -> None:
    """
    Write generation log data to a text file.

    Args:
        log_data: Dictionary containing prompt and generation data
        log_file: Path to output log file
    """
    with open(log_file, 'w') as f:
        for entry_num, entry in enumerate(log_data, 1):
            # Write prompt section
            f.write("###### ORIGINAL PROMPT #####\n\n")
            f.write(entry['prompt']['text'] + "\n\n")
            f.write("#### ANS ####\n\n")
            f.write(str(entry['prompt']['answer']) + "\n\n")

            # Write each generation
            for i, gen in enumerate(entry['generations'], 1):
                f.write(f"#### GENERATION {i} RESPONSE ####\n\n")
                f.write(gen['response'] + "\n\n")
                f.write(f"#### GENERATION {i} SCORES ####\n")
                
                # Write individual scores
                f.write(f"Correctness: {gen['scores']['correctness']}\n")
                f.write(f"Integer format: {gen['scores']['integer_format']}\n") 
                f.write(f"Strict format: {gen['scores']['strict_format']}\n")
                f.write(f"Soft format: {gen['scores']['soft_format']}\n")
                f.write(f"XML count: {gen['scores']['xml_count']}\n")
                f.write(f"Total reward: {gen['scores']['total_reward']}\n\n")

def _selective_log_softmax(
    logits: Float[Array, "B T V"],
    predicted_tokens: Int[Array, "B T"],
) -> Float[Array, "B T"]:
    if logits.dtype in [jnp.float32, jnp.float64]:
        selected_logits = jnp.take_along_axis(logits, indices=predicted_tokens[..., None], axis=-1).squeeze(-1)
        logsumexp_values = jnp.stack([jax.nn.logsumexp(lg, axis=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, predicted_tokens):
            row_logps = jax.nn.log_softmax(row_logits, axis=-1)
            row_per_token_logps = jnp.take_along_axis(row_logps, indices=row_labels[..., None], axis=-1).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = jnp.stack(per_token_logps)
    return per_token_logps

def _get_per_token_logps(
   model: qwenjax.Weights,
   input_ids: jax.Array,
   attention_mask: jax.Array,
   position_ids: jax.Array,
   cfg: qwenjax.Config,
   logits_to_keep: int, 
):
    logits, _ = qwenjax.forward(
        x=input_ids,
        weights=model,
        segment_pos=position_ids,
        attn_mask=attention_mask,
        cfg=cfg,
        cache=None,
    )
    # Keep the `logits_to_keep` logits before the final token
    logits = logits[:, -logits_to_keep-1:-1, :]
    input_ids = input_ids[:, -logits_to_keep:]
    return _selective_log_softmax(
        logits=logits,
        predicted_tokens=input_ids,
    )