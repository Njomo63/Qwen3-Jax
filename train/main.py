from .grpo import GRPO, compute_loss
import qwen.model as qwenjax
from . import rldatasets 
from . import utils 
from . import evaluator

from functools import partial
from etils import epath
import argparse
from tqdm import tqdm
import os
from typing import Tuple, List

import jax
import optax
import orbax.checkpoint as ocp

def parse_args():
    parser = argparse.ArgumentParser(description="GRPO Training Script")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen3-0.6B-Base", help="Name of base model")
    parser.add_argument("--model_path", type=str, default="/model", help="Path of base model")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset to use for training")
    
    # grpo parameters
    parser.add_argument("--cache_length", type=int, default=1024, help="Cache length for GRPO")
    parser.add_argument("--evaluator", type=str, default="gsm8k", help="Evaluator for scoring")
    parser.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=-1, help="Top-k sampling parameter")
    parser.add_argument("--min_p", type=float, default=0.05, help="Min-p sampling parameter")
    parser.add_argument("--num_generations", type=int, default=6, help="Number of generations")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens to generate")
    parser.add_argument("--epsilon", type=float, default=0.1, help="Epsilon for GRPO")
    parser.add_argument("--beta", type=float, default=0.04, help="Beta for GRPO")
    parser.add_argument("--use_clipping", action="store_true", help="Use clipping for GRPO")
    
    # Optimizer parameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for optimizer")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_percent", type=float, default=0.18, help="Percentage of total steps for warmup")
    
    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")

    # Checkpointing
    parser.add_argument("--checkpoint_save_freq", type=int, default=100, help="How often to save checkpoints")
    parser.add_argument("--keep_last_n_checkpoints", type=int, default=5, help="How many checkpoints to keep")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size")
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Number of training iterations")    
    parser.add_argument("--update_ref_model_freq", type=int, default=200, help="How often to update reference model")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    return args

def build_grpo_trainer(model, cfg, tokenizer, args, rng):
    eval_class = evaluator.get_evaluator(args.evaluator)
    return GRPO(
        model=model,
        cfg=cfg,
        tokenizer=tokenizer,
        cache_length=args.cache_length,
        evaluator=eval_class,
        rng=rng,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        num_iterations=args.num_iterations,
        epsilon=args.epsilon,
        beta=args.beta,
        use_clipping=args.use_clipping
    )
    
if __name__ == "__main__":
    args = parse_args()
    model_path = epath.Path(args.model_path)
    
    # Setup logging directory
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)
    
    # Setup checkpoint directory
    checkpoint_dir = os.path.join(args.output_dir, f"{args.model_name}_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        options=ocp.CheckpointManagerOptions(
            save_interval_steps=args.checkpoint_save_freq,
            max_to_keep=args.keep_last_n_checkpoints,
            step_prefix="round_step"
        )
    )  
    
    # Setup optimizer
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    warmup_steps = max(1, warmup_steps)
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=args.learning_rate,
        transition_steps=warmup_steps,
    )
    decay_schedule = optax.cosine_decay_schedule(
        init_value=args.learning_rate,
        decay_steps=args.num_train_iters - warmup_steps,
    )
    learning_rate_schedule = optax.join_schedules(
        schedules=[warmup_schedule, decay_schedule],
        boundaries=[warmup_steps]
    )
    base_optimizer = optax.adamw(
        learning_rate=learning_rate_schedule,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    optimizer_with_clip = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        base_optimizer
    )
    optimizer = optax.MultiSteps(
        opt=optimizer_with_clip,
        every_k_schedule=args.gradient_accumulation_steps,
    )
    
    @partial(jax.jit, static_argnames=["cfg", "use_clipping", "beta", "epsilon"])
    def train_step( 
        model: qwenjax.Weights,
        opt_state: optax.OptState, 
        inputs: Tuple[List[str], List[str]],
        cfg: qwenjax.Config,
        use_clipping: bool,
        beta: float,
        epsilon: float
    ):
        (loss, metrics), grads = jax.value_and_grad(compute_loss, has_aux=True)(model, inputs, cfg, use_clipping, beta, epsilon)
        updates, new_opt_state = optimizer.update(grads, opt_state, model)
        new_model = optax.apply_updates(model, updates)      
        return new_model, new_opt_state, loss, metrics
    
    # --- Setup initial states for training (or restore from checkpoint) ---
    tokenizer = qwenjax.load_tokenizer(model_path)
    cfg = qwenjax.load_config(model_path / "config.json")
    model = qwenjax.load_pytree(model_path, qwenjax.Weights.init_placeholder(cfg))
    opt_state = optimizer.init(model)
    rng = jax.random.PRNGKey(args.seed)
    start_round_num = 0
    restored_target = {
        'model': model,
        'opt_state': opt_state,
        'round_num': 0,
        'rng': rng,
    }
    latest_step_in_manager = checkpoint_manager.latest_step()
    if latest_step_in_manager is not None:
        print(f"Resuming training from checkpoint at step {latest_step_in_manager}...")
        restored_items = checkpoint_manager.restore(latest_step_in_manager, items=restored_target)
        model = restored_items['model']
        opt_state = restored_items['opt_state']
        start_round_num = restored_items['round_num']
        rng = restored_items['rng']
        
        print(f"Successfully resumed from round {start_round_num}.")
    else:
        print("No checkpoint found. Starting training from scratch.")
    # -------------- MAIN TRAINING LOOP ----------------    
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name, batch_size=args.batch_size)
    for round_num in tqdm(range(start_round_num,args.num_train_iters), 
                          desc="Training Progress",
                          initial=start_round_num):
        rng, curr_rng = jax.random.split(rng)
        grpo_trainer = build_grpo_trainer(model, cfg, tokenizer, args, curr_rng) # initialize the reference model
        for update_freq_idx in tqdm(range(args.update_ref_model_freq),
                                    desc=f"Ref Model Update Cycle (Round {round_num})",
                                    leave=False):
            prompts, answers = next(train_loader)
            inputs, metrics, log_data = grpo_trainer._prepare_inputs(prompts, answers)
            log_file = os.path.join(train_log_dir, f'{round_num}_generations.txt')
            utils.write_generation_log(log_data, log_file)
            pbar = tqdm(range(grpo_trainer.num_iterations), desc=f"GRPO Iterations (Batch {update_freq_idx})", leave=False)
            for i in pbar:
                model, opt_state, loss, metrics = train_step(model, opt_state, inputs, grpo_trainer.cfg, grpo_trainer.use_clipping, grpo_trainer.beta, grpo_trainer.epsilon)
                pbar.set_postfix(loss=f"{float(loss):.4f}", mean_kl=f"{float(metrics.get('mean_kl', -1.0)):.4f}") 

        # --- Checkpoint Saving ---
        if (round_num + 1) % args.checkpoint_save_freq == 0 or (round_num + 1) == args.num_train_iters:
            print(f"Saving checkpoint at round {round_num}...")
            items_to_save = {
                'model': model,
                'opt_state': opt_state,
                'rng': rng,  # Save the current state of the main RNG key
                'round_num': round_num + 1 # Save the next round number to resume from
            }
            checkpoint_manager.save(
                round_num,
                args=ocp.args.StandardSave(items_to_save) 
            )

    checkpoint_manager.wait_until_finished()
    checkpoint_manager.close()
    print("Training complete and all checkpoints saved.")  
    
    