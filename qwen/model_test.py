import pytest
import jax
import jax.numpy as jnp
from functools import partial
import dataclasses 

from qwen.model import (
    Config,
    Weights,
    KVCache,
    MLPLayer,      
    AttentionLayer,
    Layer,
    rms_norm,
    apply_rope,
    attention_block,
    mlp_block,
    forward_layer,
    forward
)

# ---- Test Fixtures ----

@pytest.fixture(scope="module")
def key():
    """Provides a reusable JAX PRNG key."""
    return jax.random.PRNGKey(42)

@pytest.fixture(scope="module")
def base_config():
    """Provides a base configuration for testing."""
    return Config(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16, 
        max_window_layers=2, 
        vocab_size=100,
        dtype=jnp.bfloat16,
        use_cache=False,
        rope_theta=10000.0,
    )

@pytest.fixture(scope="module")
def initialized_weights(key, base_config):
    """Provides initialized weights based on the base config."""
    return Weights.init(key, base_config)


# ---- Helper Functions ----
def _get_dummy_inputs(config: Config, batch_size: int, seq_len: int):
    """Generates dummy input data for testing."""
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32) * 10 # Use token ID 10
    segment_pos = jnp.arange(seq_len, dtype=jnp.int32)[None, :] # Shape (1, L)
    segment_pos = jnp.repeat(segment_pos, batch_size, axis=0)    # Shape (B, L)

    attn_mask = jnp.tril(jnp.ones((batch_size, seq_len, seq_len), dtype=jnp.bool_))
    return input_ids, segment_pos, attn_mask

# ---- Unit Tests ----

def test_weights_initialization(initialized_weights, base_config):
    """Tests the shape and dtype of initialized weights."""
    cfg = base_config
    weights = initialized_weights

    # Check Embedding
    assert weights.embedding.shape == (cfg.vocab_size, cfg.hidden_size)
    assert weights.embedding.dtype == cfg.dtype

    # Check LM Head (when not tied)
    assert weights.lm_head.shape == (cfg.hidden_size, cfg.vocab_size)
    assert weights.lm_head.dtype == cfg.dtype

    # Check Final Norm Gamma
    assert weights.gamma_final.shape == (cfg.hidden_size,)
    assert weights.gamma_final.dtype == cfg.dtype

    # Check Layer Parameters (sanity check one layer)
    assert len(weights.layers) == cfg.max_window_layers
    layer0 = weights.layers[0]
    assert layer0.gamma_pre_attn.shape == (cfg.hidden_size,)
    assert layer0.gamma_post_attn.shape == (cfg.hidden_size,)
    assert layer0.attn.q_proj.shape == (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
    assert layer0.attn.k_proj.shape == (cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
    assert layer0.attn.v_proj.shape == (cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim)
    assert layer0.attn.o_proj.shape == (cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size)
    assert layer0.mlp.gate_proj.shape == (cfg.hidden_size, cfg.intermediate_size)
    assert layer0.mlp.up_proj.shape == (cfg.hidden_size, cfg.intermediate_size)
    assert layer0.mlp.down_proj.shape == (cfg.intermediate_size, cfg.hidden_size)

def test_rms_norm():
    """Tests the RMS normalization implementation."""
    B, L, D = 2, 5, 4
    x = jnp.ones((B, L, D), dtype=jnp.float32) * 2.0
    gamma = jnp.ones((D,), dtype=jnp.float32) * 0.5

    # Expected calculation:
    # mean_sq = mean(2.0^2) = 4.0
    # rsqrt = 1 / sqrt(4.0 + 1e-6) ~= 1 / 2.0 = 0.5
    # normed_x = x * rsqrt = 2.0 * 0.5 = 1.0
    # output = normed_x * gamma = 1.0 * 0.5 = 0.5
    expected_output = jnp.ones_like(x) * 0.5

    # Use the actual epsilon from the rms_norm function if it uses config
    # Assuming epsilon=1e-6 is hardcoded as in the provided code
    output = rms_norm(x, gamma)

    assert output.shape == x.shape
    assert output.dtype == x.dtype
    assert jnp.allclose(output, expected_output, atol=1e-6)


def test_apply_rope():
    """Tests Rotary Positional Embedding application."""
    B, L, N, H = 1, 6, 2, 4 # H must be even
    # Use a minimal config or just the values directly
    rope_theta=10000.0
    rope_scaling=1.0 # Test with a valid float
    inputs = jnp.ones((B, L, N, H), dtype=jnp.float32)
    positions = jnp.arange(L, dtype=jnp.int32)[None, :] # (1, L)

    output = apply_rope(inputs, positions, rope_theta, rope_scaling)

    assert output.shape == inputs.shape
    assert output.dtype == inputs.dtype
    # RoPE should change the values
    assert not jnp.allclose(output, inputs)
    # Check first and last positions are different
    assert not jnp.allclose(output[:, 0], output[:, -1])

    # Test with scaling != 1.0
    rope_scaling_scaled = 2.0
    output_scaled = apply_rope(inputs, positions, rope_theta, rope_scaling_scaled)
    assert output_scaled.shape == inputs.shape
    assert not jnp.allclose(output_scaled, output) # Should differ from non-scaled


def test_mlp_block(key, base_config):
    """Tests the MLP block forward pass."""
    B, L = 2, 5
    cfg = base_config
    mlp_weights = MLPLayer.init(key, cfg)
    x = jnp.ones((B, L, cfg.hidden_size), dtype=cfg.dtype)

    output = mlp_block(x, mlp_weights, cfg)

    assert output.shape == (B, L, cfg.hidden_size)
    assert output.dtype == cfg.dtype

def test_attention_block(key, base_config):
    """Tests the Attention block forward pass without cache."""
    B, L = 2, 5
    cfg = base_config 
    attn_weights = AttentionLayer.init(key, cfg)
    x = jnp.ones((B, L, cfg.hidden_size), dtype=cfg.dtype)
    _, segment_pos, attn_mask = _get_dummy_inputs(cfg, B, L) # Mask shape (B, L, L)

    # Test without cache first
    output, cache_updates = attention_block(x, segment_pos, attn_mask, attn_weights, cfg, idx=0, cache=None)

    assert output.shape == (B, L, cfg.hidden_size)
    assert output.dtype == cfg.dtype
    assert cache_updates is None

def test_attention_block_with_cache(key, base_config):
    """Tests the Attention block forward pass with cache."""
    B, L = 2, 3 
    S = 10 # Cache size

    # --- FIX 2: Correctly create config with use_cache=True ---
    config_dict = dataclasses.asdict(base_config)
    config_dict['use_cache'] = True
    cfg = Config(**config_dict) 
    # ----------------------------------------------------------

    attn_weights = AttentionLayer.init(key, cfg)
    init_key, cache_key = jax.random.split(key)
    x = jax.random.normal(init_key, (B, L, cfg.hidden_size), dtype=cfg.dtype)
    _, segment_pos, _ = _get_dummy_inputs(cfg, B, L)

    attn_mask = jnp.ones((B, L, S), dtype=jnp.bool_)

    # Initialize cache
    cache = KVCache.init(cache_key, cfg, batch_size=B, cache_size=S, dtype=cfg.dtype)
    initial_k_cache = cache.k_cache[0].copy() # Copy initial state for comparison
    initial_v_cache = cache.v_cache[0].copy()

    # Need to ensure cache is passed correctly and idx is valid
    output, cache_updates = attention_block(x, segment_pos, attn_mask, attn_weights, cfg, idx=0, cache=cache)
    updated_k, updated_v = cache_updates

    assert output.shape == (B, L, cfg.hidden_size)
    assert output.dtype == cfg.dtype
    assert cache_updates is not None
    assert updated_k.shape == (B, S, cfg.num_key_value_heads, cfg.head_dim)
    assert updated_v.shape == (B, S, cfg.num_key_value_heads, cfg.head_dim)
    assert updated_k.dtype == cfg.dtype
    assert updated_v.dtype == cfg.dtype

    # Check that *some* part of the cache was updated (not rigorously checking content)
    assert not jnp.all(updated_k == initial_k_cache)
    assert not jnp.all(updated_v == initial_v_cache)


def test_forward_layer(key, base_config):
    """Tests a single transformer layer forward pass."""
    B, L = 2, 5
    cfg = base_config 
    layer_weights = Layer.init(key, cfg)
    x = jnp.ones((B, L, cfg.hidden_size), dtype=cfg.dtype)
    _, segment_pos, attn_mask = _get_dummy_inputs(cfg, B, L)

    output, cache_updates = forward_layer(x, segment_pos, attn_mask, layer_weights, cfg, idx=0, cache=None)

    assert output.shape == (B, L, cfg.hidden_size)
    assert output.dtype == cfg.dtype
    assert cache_updates is None # Cache not used here

# # (Keep test_forward_pass_shapes and test_forward_pass_with_cache as is,
# # they depend on the base_config fix propagating correctly)
# @pytest.mark.parametrize("tie_embeddings", [True, False])
# def test_forward_pass_shapes(initialized_weights, initialized_weights_tied, base_config, tie_embeddings):
#     """Tests the full forward pass for shape and dtype, checking weight tying."""
#     B, L = 2, 7

#     # Create config based on tie_embeddings flag
#     config_dict = dataclasses.asdict(base_config)
#     config_dict['tie_word_embeddings'] = tie_embeddings
#     cfg = Config(**config_dict)

#     weights = initialized_weights_tied if tie_embeddings else initialized_weights

#     input_ids, segment_pos, attn_mask = _get_dummy_inputs(cfg, B, L) # Mask shape (B, L, L)

#     # Test without cache
#     logits, final_cache = forward(input_ids, segment_pos, attn_mask, weights, cfg, cache=None)

#     assert logits.shape == (B, L, cfg.vocab_size)
#     # Logits are often float32 even with bfloat16 intermediate for stability
#     assert logits.dtype == jnp.float32 or logits.dtype == cfg.dtype # Allow either possibility
#     assert final_cache is None

# @pytest.mark.parametrize("tie_embeddings", [True, False])
# def test_forward_pass_with_cache(key, initialized_weights, initialized_weights_tied, base_config, tie_embeddings):
#     """Tests the full forward pass with KV caching enabled."""
#     B, L = 1, 1 # Simulate single token generation
#     cache_len = 10

#     # Create config based on flags
#     config_dict = dataclasses.asdict(base_config)
#     config_dict['tie_word_embeddings'] = tie_embeddings
#     config_dict['use_cache'] = True
#     cfg = Config(**config_dict) # This has rope_scaling=1.0, correct tie_..., use_cache=True

#     weights = initialized_weights_tied if tie_embeddings else initialized_weights

#     init_key, cache_key = jax.random.split(key)
#     input_ids, segment_pos, _ = _get_dummy_inputs(cfg, B, L) # L=1
#     # Mask shape for cache: (B, L_query, L_memory) -> (B, 1, cache_len)
#     # Allow attending to all cache positions initially for simplicity
#     attn_mask = jnp.ones((B, L, cache_len), dtype=jnp.bool_)

#     cache = KVCache.init(cache_key, cfg, batch_size=B, cache_size=cache_len, dtype=cfg.dtype)
#     initial_end_index = cache.end_index[0].copy()

#     logits, updated_cache = forward(input_ids, segment_pos, attn_mask, weights, cfg, cache=cache)

#     assert logits.shape == (B, L, cfg.vocab_size)
#     assert updated_cache is not None
#     assert isinstance(updated_cache, KVCache) # Check it's the right type
#     assert len(updated_cache.k_cache) == cfg.max_window_layers
#     assert updated_cache.k_cache[0].shape == (B, cache_len, cfg.num_key_value_heads, cfg.head_dim)
#     assert updated_cache.v_cache[0].shape == (B, cache_len, cfg.num_key_value_heads, cfg.head_dim)
#     # Check end index was updated (assuming the mutable update in forward works outside JIT)
#     assert updated_cache.end_index[0].shape == (B,)
#     assert updated_cache.end_index[0][0] == initial_end_index[0] + L