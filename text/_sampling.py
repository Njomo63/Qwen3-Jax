"""Sampling Class."""
import dataclasses

import jax
import jax.numpy as jnp

@dataclasses.dataclass(frozen=True, kw_only=True)    
class CombinedSampling:
  top_k: int = -1  # -1 means no top-k filtering
  top_p: float = 1.0  # 1.0 means no top-p filtering
  min_p: float = 0.0  # Minimum probability threshold. 0.0 means no filtering.
  temperature: float = 1.0  # Temperature for scaling logits
  
  def __post_init__(self):
    self._verify_args()
    if self.temperature == 0.0:
      # Zero temperature means greedy sampling.
      object.__setattr__(self, 'top_p', 1.0)
      object.__setattr__(self, 'top_k', -1)
      object.__setattr__(self, 'min_p', 0.0)

  def _verify_args(self):
    if self.temperature < 0.0:
      raise ValueError(
          f"temperature must be non-negative, got {self.temperature}.")
    if not 0.0 < self.top_p <= 1.0:
      raise ValueError(f"top_p must be in (0, 1], got {self.top_p}.")
    if self.top_k < -1 or self.top_k == 0:
      raise ValueError(f"top_k must be -1 (disable), or at least 1, "
                       f"got {self.top_k}.")
    if not isinstance(self.top_k, int):
      raise TypeError(
        f"top_k must be an integer, got {type(self.top_k).__name__}")
    if not 0.0 <= self.min_p <= 1.0:
      raise ValueError("min_p must be in [0, 1], got "
                       f"{self.min_p}.")
    
  def sample(self, logits, rng):
    if self.temperature != 1.0:
      logits = logits / self.temperature
    
    if self.min_p != 0.0:
      logits = self._apply_min_p(logits)
      
    if self.top_k != -1:
      logits = self._apply_top_k(logits)
      
    if self.top_p < 1.0:
      logits = self._apply_top_p(logits)
      
    if self.temperature == 0.0:
      # Greedy sampling
      return jnp.argmax(logits, axis=-1)
    else:
      return jax.random.categorical(rng, logits, axis=-1)
  
  def _apply_min_p(self, logits):
    probs = jax.nn.softmax(logits, axis=-1)
    top_probs = jnp.max(probs, axis=-1, keepdims=True)
    threshold = self.min_p * top_probs
    mask = probs < threshold
    return jnp.where(mask, -jnp.inf, logits) 
  
  def _apply_top_k(self, logits): 
    topk_values, _ = jax.lax.top_k(logits, self.top_k)
    kth_value = topk_values[..., -1:]
    mask = logits < kth_value
    return jnp.where(mask, -jnp.inf, logits)  
  
  def _apply_top_p(self, logits):
    probs = jax.nn.softmax(logits, axis=-1)
    sorted_probs, sorted_indices = jax.lax.top_k(probs, probs.shape[-1])

    cumsum = jnp.cumsum(sorted_probs, axis=-1)
    mask = cumsum - sorted_probs > self.top_p

    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
    filtered_sorted_logits = jnp.where(mask, -jnp.inf, sorted_logits)
    original_indices = jnp.argsort(sorted_indices, axis=-1)
    filtered_logits = jnp.take_along_axis(filtered_sorted_logits, original_indices, axis=-1)
    return filtered_logits

      
    