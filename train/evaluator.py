import re
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from jaxtyping import Array, Float, Int

class RewardEvaluator(ABC):
    """
    Abstract base class for reward computation in RL training.
    
    This class defines the interface for reward evaluators that can be used
    to score model completions during RL training. Implement this class to
    create custom reward functions for different tasks.
    
    The main methods that need to be implemented are:
    - compute_rewards: Computes rewards for a batch of completions
    - get_reward_breakdown: Converts raw reward scores to a labeled dictionary
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        completions: List[List[Dict[str, str]]],
        answer: Any,
    ) -> Tuple[jax.Array, Dict[str, float]]:
        """
        Compute rewards for a batch of completions.
        
        Args:
            completions: List of completion messages in chat format
                        [{"role": "assistant", "content": "..."}, ...]
            answer: Ground truth answer(s) for the prompts
            
        Returns:
            rewards_per_func: Array of shape (num_completions, num_reward_functions)
                            containing individual reward function scores
            metrics: Dictionary of aggregated metrics including mean rewards
                    per function and total reward
        """
        raise NotImplementedError("compute_rewards must be implemented in subclasses")
    
    @abstractmethod
    def get_reward_breakdown(self, reward_scores: jax.Array) -> Dict[str, float]:
        """
        Convert raw reward scores array to a labeled dictionary.
        
        Args:
            reward_scores: Array of raw scores from compute_rewards
            
        Returns:
            Dictionary mapping reward function names to their scores
        """
        raise NotImplementedError("get_reward_breakdown must be implemented in subclasses")
    
def get_evaluator(name: str) -> RewardEvaluator:
    """
    Get the appropriate reward evaluator for a given task.
    
    Args:
        name: Name of the task/dataset to get evaluator for
        
    Returns:
        RewardEvaluator instance for the specified task
        
    Raises:
        NotImplementedError: If evaluator for given task is not implemented
    """
    if name.lower() == "gsm8k":
        return GSM8kEvaluator()
    else:
        raise NotImplementedError(f"No evaluator implemented for {name}")
    
class GSM8kEvaluator(RewardEvaluator):
    """
    Reward evaluator for the GSM8K math problem dataset.
    
    Implements reward functions for:
    - Answer correctness
    - Integer format validation
    - XML formatting (strict and soft)
    - XML tag counting
    """
    
    def __init__(self):
        self.num_reward_functions = 5
    
    def _extract_xml_answer(self, text: str) -> str:
        """Extract answer from XML tags."""
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    
    def _correctness_reward(self, completions, answers) -> List[float]:
        """Reward for correct answer."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [2.0 if r == a else 0.0 for r, a in zip(extracted, answers)]
    
    def _int_format_reward(self, completions) -> List[float]:
        """Reward for integer format."""
        responses = [completion[0]['content'] for completion in completions]
        extracted = [self._extract_xml_answer(r) for r in responses]
        return [0.5 if r.isdigit() else 0.0 for r in extracted]
    
    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for strict XML format."""
        pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.match(pattern, r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]
    
    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for relaxed XML format."""
        pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [bool(re.match(pattern, r)) for r in responses]
        return [0.5 if m else 0.0 for m in matches]
    
    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for XML tag counting."""
        def count_xml(text: str) -> float:
            count = 0.0
            if text.count("<reasoning>\n") == 1: count += 0.125
            if text.count("\n</reasoning>\n") == 1: count += 0.125
            if text.count("\n<answer>\n") == 1:
                count += 0.125
                count -= len(text.split("\n</answer>\n")[-1])*0.001
            if text.count("\n</answer>") == 1:
                count += 0.125
                count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
            return count
            
        responses = [completion[0]["content"] for completion in completions]
        return [count_xml(r) for r in responses]
    
    def compute_rewards(
        self,
        completions: List[List[Dict[str, str]]],
        answers: Any,
    ) -> Tuple[Float[Array, "num_completions num_rewards"], Dict[str, float]]:
        """Compute all rewards for the given completions."""
        num_completions = len(completions)
        rewards_per_func = jnp.zeros((num_completions, self.num_reward_functions))  # shape: (num_completions, num_reward_functions)
        
        # Compute all reward functions
        all_scores = [
            self._correctness_reward(completions, answers),
            self._int_format_reward(completions),
            self._strict_format_reward(completions),
            self._soft_format_reward(completions),
            self._xml_count_reward(completions)
        ]
        
        # Fill rewards array
        for i, scores in enumerate(all_scores):
            rewards_per_func = rewards_per_func.at[:, i].set(jnp.array(scores, dtype=jnp.float32))
        
        # Compute metrics
        reward_per_func = rewards_per_func.mean(axis=0)
        
        # Calculate accuracy (perfect correctness score)
        correctness_scores = rewards_per_func[:, 0]  # First reward function is correctness
        num_perfect = (correctness_scores == 2.0).sum().item()
        accuracy = num_perfect / num_completions
        
        metrics = {
            "rewards/correctness_reward_func": reward_per_func[0].item(),
            "rewards/int_reward_func": reward_per_func[1].item(), 
            "rewards/strict_format_reward_func": reward_per_func[2].item(),
            "rewards/soft_format_reward_func": reward_per_func[3].item(),
            "rewards/xmlcount_reward_func": reward_per_func[4].item(),
            "reward": rewards_per_func.sum(axis=1).mean().item(),
            "accuracy": accuracy
        }
        return rewards_per_func, metrics
    
    def get_reward_breakdown(self, reward_scores: jax.Array) -> Dict[str, float]:
        """Convert reward scores tensor to labeled dictionary."""
        return {
            'correctness': reward_scores[0].item(),
            'integer_format': reward_scores[1].item(),
            'strict_format': reward_scores[2].item(),
            'soft_format': reward_scores[3].item(),
            'xml_count': reward_scores[4].item()
        }
    
    