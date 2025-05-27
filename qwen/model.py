import dataclasses
from functools import partial
from dataclasses import field
from typing import Callable
from pathlib import Path
from etils import epath
import os

import jax
import jax.numpy as jnp
from jax import tree_util, ShapeDtypeStruct
from jaxtyping import Array, Float, Bool, Int
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoConfig


@tree_util.register_static
@dataclasses.dataclass(kw_only=True, frozen=True)
class Config:
    attention_bias: bool
    attention_dropout: float
    bos_token_id: int 
    eos_token_id: int 
    head_dim: int 
    hidden_act: str 
    hidden_size: int
    initializer_range: float
    intermediate_size: int 
    max_position_embeddings: int 
    max_window_layers: int 
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_scaling: None
    rope_theta: int
    sliding_window : int | None
    tie_word_embeddings: bool
    use_cache: bool 
    use_sliding_window: bool 
    vocab_size: int 
    dtype: jnp.dtype = jnp.bfloat16

def hf_to_jax_config(qwen_config) -> Config:
    _get = lambda x, k, default=None: getattr(x, k, default) if hasattr(x, k) else dict(x).get(k, default)
    return Config(
        attention_bias=_get(qwen_config, "attention_bias"),
        attention_dropout=_get(qwen_config, "attention_dropout"),
        bos_token_id=_get(qwen_config, "bos_token_id"),
        eos_token_id=_get(qwen_config, "eos_token_id"),
        head_dim= _get(qwen_config, "head_dim"),
        hidden_act=_get(qwen_config, "hidden_act"),
        hidden_size=_get(qwen_config, "hidden_size"),
        initializer_range=_get(qwen_config, "initializer_range"),
        intermediate_size=_get(qwen_config, "intermediate_size"),
        max_position_embeddings=_get(qwen_config, "max_position_embeddings"),
        max_window_layers=_get(qwen_config, "max_window_layers"),
        num_attention_heads=_get(qwen_config, "num_attention_heads"),
        num_key_value_heads=_get(qwen_config, "num_key_value_heads"),
        rms_norm_eps=_get(qwen_config, "rms_norm_eps"),
        rope_scaling=_get(qwen_config, "rope_scaling"),
        rope_theta=_get(qwen_config, "rope_theta"),
        sliding_window=_get(qwen_config, "sliding_window"),
        tie_word_embeddings=_get(qwen_config, "tie_word_embeddings"),
        use_cache=_get(qwen_config, "use_cache"),
        use_sliding_window=_get(qwen_config, "use_sliding_window"),
        vocab_size=_get(qwen_config, "vocab_size"),
    )

def load_config(config_path: str | os.PathLike[str] | Path) -> "Config":
    # return llama_to_jax_config(json.loads(Path(config_path).read_text()))
    return hf_to_jax_config(AutoConfig.from_pretrained(config_path))

def load_tokenizer(
    tokenizer_path: str | os.PathLike[str] | Path
) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(tokenizer_path)

 
def jax_pytree_struct(cls, meta_fields: tuple = ()):
    """jax.tree_util.register_dataclass wrapper that automatically infers data_fields."""
    assert not dataclasses.is_dataclass(cls)
    cls = dataclasses.dataclass(cls)
    all_fields = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    data_fields = tuple(f for f in all_fields if f not in meta_fields)
    return tree_util.register_dataclass(cls, data_fields=data_fields, meta_fields=meta_fields)

@partial(jax_pytree_struct, meta_fields=("shape", "dtype", "initializer", "metadata"))
class ArrayInfo:
    shape: tuple[int, ...]
    dtype: "jnp.dtype"
    initializer: Callable | None = None
    metadata: dict = field(default_factory=dict)

is_type = lambda x, cls: (type(x).__name__ == cls.__name__) and (type(x).__module__ == cls.__module__)
is_param = lambda x: is_type(x, ArrayInfo)

class _Init:
    """Base class for pytree data structures that will eventually contain jax.Arrays (e.g. layer definitions).
    Each subclass is responsible for defining abstract(), which returns an "abstract" version of the pytree containing
    ArrayInfos (i.e. metadata) instead of actual data. This class then helps generate the actual data.
    """
    @classmethod
    def abstract(cls, cfg: Config, *args, **kw):
        """Returns an instance of this class with ArrayInfos instead of jax.Arrays."""
        raise NotImplementedError

    @classmethod
    def init(cls, key: jax.random.PRNGKey, cfg: Config, *args, **kw):
        """Returns a pytree of randomly-initialized jax.Arrays corresponding to abstract()."""
        abstract = cls.abstract(cfg, *args, **kw)

        @jax.jit
        def _init():
            num_leaves = len(jax.tree.leaves(abstract, is_leaf=lambda x: isinstance(x, ArrayInfo)))
            key_iter = iter(jax.random.split(key, num_leaves))

            def init_leaf(x):
                if isinstance(x, ArrayInfo):
                    return x.initializer(next(key_iter), x.shape, x.dtype)
                return x

            return jax.tree.map(
                init_leaf,
                abstract,
                is_leaf=lambda x: isinstance(x, ArrayInfo),
            )
        return _init()
    
@jax_pytree_struct
class MLPLayer(_Init):
    gate_proj: jax.Array | ArrayInfo
    up_proj: jax.Array | ArrayInfo
    down_proj: jax.Array | ArrayInfo
    
    @classmethod
    def abstract(cls, cfg: Config):
        _init = jax.nn.initializers.truncated_normal(cfg.initializer_range)
        dtype = cfg.dtype
        layer = MLPLayer(
            gate_proj = ArrayInfo((cfg.hidden_size, cfg.intermediate_size), dtype, _init),
            up_proj = ArrayInfo((cfg.hidden_size, cfg.intermediate_size), dtype, _init),
            down_proj = ArrayInfo((cfg.intermediate_size, cfg.hidden_size), dtype, _init),
        )
        return layer
    
@jax_pytree_struct
class AttentionLayer(_Init):
    q_proj: jax.Array | ArrayInfo
    k_proj: jax.Array | ArrayInfo
    v_proj: jax.Array | ArrayInfo
    o_proj: jax.Array | ArrayInfo
    q_norm: jax.Array | ArrayInfo
    k_norm: jax.Array | ArrayInfo
    
    @classmethod
    def abstract(cls, cfg: Config):
        _init = jax.nn.initializers.truncated_normal(cfg.initializer_range)
        _norm_init = jax.nn.initializers.constant(1.0)
        dtype = cfg.dtype
        layer = AttentionLayer(
            q_proj = ArrayInfo((cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
                               dtype,
                                _init),
            k_proj = ArrayInfo((cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
                               dtype, 
                               _init),
            v_proj = ArrayInfo((cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
                               dtype,
                               _init),
            o_proj = ArrayInfo((cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
                               dtype, 
                               _init),
            q_norm = ArrayInfo((cfg.head_dim,), dtype, _norm_init),
            k_norm = ArrayInfo((cfg.head_dim,), dtype, _norm_init),
        )
        return layer
    
@jax_pytree_struct
class Layer(_Init):
    mlp: MLPLayer
    attention: AttentionLayer
    input_layernorm: jax.Array | ArrayInfo
    post_attention_layernorm: jax.Array | ArrayInfo
    
    @classmethod
    def abstract(cls, cfg: Config) -> "Layer":
        _init = jax.nn.initializers.constant(1.0)
        dtype = cfg.dtype
        return Layer(
            mlp = MLPLayer.abstract(cfg),
            attention = AttentionLayer.abstract(cfg),
            input_layernorm = ArrayInfo((cfg.hidden_size,), dtype, _init),
            post_attention_layernorm = ArrayInfo((cfg.hidden_size,), dtype, _init)
        )
        
@jax_pytree_struct
class Weights(_Init):
    layers: list[Layer]
    embedding: jax.Array | ArrayInfo
    final_norm: jax.Array | ArrayInfo
    
    @classmethod
    def abstract(cls, cfg: Config):
        layers = [Layer.abstract(cfg) for _ in range(cfg.max_window_layers)]
        return Weights(
            layers = layers,
            embedding = ArrayInfo(
                (cfg.vocab_size, cfg.hidden_size),
                cfg.dtype,
                jax.nn.initializers.truncated_normal(cfg.initializer_range)
            ),
            final_norm = ArrayInfo(
                (cfg.hidden_size,),
                cfg.dtype,
                jax.nn.initializers.constant(1.0)
            ),
        )
    @classmethod
    def init_placeholder(cls, cfg):
        abstract = cls.abstract(cfg)
        return jax.tree.map(
            lambda info: ShapeDtypeStruct(shape=info.shape, dtype=info.dtype),
            abstract,
            is_leaf=is_param,
        )

@jax_pytree_struct  
class KVCache(_Init):
    k_cache: list[jax.Array]
    v_cache: list[jax.Array]
    end_index: list[jax.Array]
    
    @classmethod
    def abstract(cls, cfg: Config, batch_size: int, cache_size: int, dtype: jnp.dtype = jnp.bfloat16):
        _init = jax.nn.initializers.zeros
        # The cache shape is (batch_size, cache_size, num_heads, head_dim)
        cache_shape = (batch_size, cache_size, cfg.num_key_value_heads, cfg.head_dim)
        k_cache_info = ArrayInfo(cache_shape, dtype, _init)
        v_cache_info = ArrayInfo(cache_shape, dtype, _init)
        end_index_info = ArrayInfo((batch_size,), jnp.int32, _init)
        
        cache = KVCache(
            k_cache = [k_cache_info for _ in range(cfg.max_window_layers)],
            v_cache = [v_cache_info for _ in range(cfg.max_window_layers)],
            end_index = [end_index_info for _ in range(cfg.max_window_layers)],
        )
        return cache

def rms_norm(
    x: Float[Array, "B L D"],
    gamma: Float[Array, "D"],
) -> Float[Array, "B L D"]:
    """Applies RMS normalization.
    """
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    normed_inputs = x * jax.lax.rsqrt(var + 1e-06)
    
    gamma = jnp.expand_dims(gamma, axis=range(len(x.shape) - 1))
    normed_inputs = normed_inputs * gamma
    return normed_inputs
    
    
def apply_rope(
    inputs: Float[Array, "B L N H"],
    positions: Int[Array, "B L"],
    base_frequency: int,
    scale_factor: float | None = None,
) -> Float[Array, "B L N H"]:
    """Applies RoPe
    Let B denote batch size, L denote sequence length, N denote number of heads, 
    and H denote head dimension.
    Args:
        inputs: Array of shape [B, L, N, H].
        positions:  Array of shape [B, L].
        base_frequency: Base frequency used to compute rotations.
        scale_factor: The scale factor used for positional interpolation, allowing 
            an expansion of sequence length beyond the pre-trained context length.

    Returns:
        Array of shape [B, L, N, H].
    """
    head_dim = inputs.shape[-1]
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = base_frequency**fraction
    sinusoidal_inp = (
        positions[..., jnp.newaxis] / timescale[jnp.newaxis, jnp.newaxis, :]
    )
    sinusoidal_inp = sinusoidal_inp[..., jnp.newaxis, :]
    if scale_factor is not None:
        if scale_factor < 1.0:
            raise ValueError(f'scale_factor must be >= 1.0, got {scale_factor}')
        sinusoidal_inp /= scale_factor
    
    sin = jnp.sin(sinusoidal_inp)
    cos = jnp.cos(sinusoidal_inp)
    
    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = jnp.concatenate([first_part, second_part], axis=-1)
    return out.astype(inputs.dtype)


def _create_sliding_mask(
    segment_pos: Int[Array, "B T"],
    end_index: int,
    cache_len: int,
    sliding_window_size: int,
) -> Bool[Array, "B T cache_len"]:
  """Creates mask for sliding window attention.
    Args:
    segment_pos: JAX array of shape `[B, T]` containing the position IDs
      for the current query segment. `B` is batch size, `T` is the
      current segment length.
    end_index: The index where the next token would be written in the cache
    cache_len: The total length of the source sequence 
      over which to compute the mask and reconstruct absolute positions.
    sliding_window_size: The size of the attention window. 
    
    Returns:
    A boolean JAX array of shape `[B, T, S]` (where `S` is the value passed
    as `cache_len`)
  """
  total_tokens = end_index + segment_pos.shape[1]  # cached + processing tokens

  def _reconstruct_rotated_cache_positions():
    cache_positions = jnp.arange(cache_len) + total_tokens - cache_len
    cache_positions = (
        jnp.zeros_like(cache_positions)
        # kv were placed at index (position_id % cache_len) in the cache.
        .at[cache_positions % cache_len].set(cache_positions)
    )
    return cache_positions

  # Reconstruct position_ids for cached kv.
  cache_positions = jax.lax.cond(
      total_tokens <= cache_len,
      lambda: jnp.arange(cache_len),
      _reconstruct_rotated_cache_positions,
  )

  cache_positions = cache_positions[None, None, :]  # [1, 1, cache_len]
  segment_pos = segment_pos[:, :, None]  # [B, seq_len, 1]
  sliding_mask = cache_positions > segment_pos - sliding_window_size
  sliding_mask *= cache_positions < segment_pos + sliding_window_size
  return sliding_mask


def _get_attn_scale(q_head_dim: int):
    return q_head_dim**-0.5

def attention(
    q_proj: Float[Array, "B T N H"],
    k_proj: Float[Array, "B S K H"],
    v_proj: Float[Array, "B S K H"],
    segment_pos: Int[Array, "B T"],
    end_index: int,
    attn_mask: Bool[Array, "B T S"],
    cfg: Config
) -> Float[Array, "B T NH"]:
    """Computes multi-head (or grouped-query) attention output.
        **Dimension Symbols Used Below:**
        B: Batch size
        T: Target sequence length (number of queries, length of `q`)
        S: Source sequence length (number of keys/values, length of `k` & `v`)
        Note: `S` may differ from `T` due to KV caching.
        N: Number of query attention heads (from `cfg.num_attention_heads`)
        K: Number of key/value attention heads (from `cfg.num_key_value_heads`)
        - K = N for standard Multi-Head Attention (MHA)
        - 1 <= K < N for Grouped-Query/Multi-Query Attention (GQA/MQA)
        H: Dimension of each attention head (e.g., `q.shape[-1]`)

    This function implements scaled dot-product attention, supporting
    standard Multi-Head Attention (MHA), Grouped-Query Attention (GQA),
    and Multi-Query Attention (MQA). It handles causal masking (decoder-style),
    padding masking, and optional sliding window attention based on the
    provided configuration (`cfg`) and inputs.
    Args:
        q_proj: Array of shape [B, T, N, H]
        k_proj: Array of shape [B, S, K, H]
        v_proj: Array of shape [B, S, K, H]
        segment_pos: Array of shape [B, T] Position IDs for the query sequence
        segment_ids: Array of shape [B, T] with 1s for non-padded tokens and 0s for padded tokens.
        end_index: total number of processed tokens
        cfg (Config): config object

    Returns:
        Array of shape [B, T, NH] with the attention output.
    """
    scale = _get_attn_scale(q_proj.shape[-1])
    q_proj = q_proj * scale
    use_gqa = cfg.num_attention_heads != cfg.num_key_value_heads and cfg.num_key_value_heads > 1    
    if use_gqa:
        b, t, kg, h = q_proj.shape
        q_proj = q_proj.reshape((b, t, cfg.num_key_value_heads, int(kg / cfg.num_key_value_heads), h))
        logits = jnp.einsum("BTKGH,BSKH->BTKGS", q_proj, k_proj)
        b, t, k, g, s = logits.shape
        logits = logits.reshape((b, t, k * g, s))
    else:
        logits = jnp.einsum('BTNH,BSNH->BTNS', q_proj, k_proj)
    if cfg.use_sliding_window:
        sliding_mask = _create_sliding_mask(
            segment_pos,
            end_index=end_index,
            # Derive cache length from attn_mask shape in case cache is None
            cache_len=attn_mask.shape[-1],
            sliding_window_size=cfg.sliding_window
        )
        # [batch_size, seq_len, cache_size]
        attn_mask *= sliding_mask
        
    # [batch_size, seq_len, num_heads, cache_size]
    padded_logits = jnp.where((jnp.expand_dims(attn_mask, -2)), logits, -1e30)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(k_proj.dtype)
    if use_gqa:
      # Reshape matrices to enable einsums over groups.
      b, t, kg, h = probs.shape
      probs = probs.reshape(
          (b, t, cfg.num_key_value_heads, int(kg / cfg.num_key_value_heads), h)
      )
      encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, v_proj)
      b, t, k, g, h = encoded.shape
      encoded = encoded.reshape((b, t, k * g, h))
    else:
        # [batch_size, seq_len, num_heads, head_dim]
        encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, v_proj)
    return encoded.reshape(q_proj.shape[0], q_proj.shape[1],-1)
    

def attention_block(
    x: Float[Array, "B L D"],
    segment_pos: Int[Array, "B L"],
    attn_mask: Bool[Array, "B L S"],
    attn_layer: AttentionLayer,
    cfg: Config,
    cache: KVCache | None,
    idx: int = 0,
) -> tuple[jax.Array, jax.Array]:
    """Attention block"""
    dtype = cfg.dtype
        
    with jax.named_scope("q_proj"):
        q_proj = jnp.einsum("bld,dr->blr", x, attn_layer.q_proj)
        q_proj = q_proj.astype(dtype)
        q_proj = q_proj.reshape(q_proj.shape[0], q_proj.shape[1], cfg.num_attention_heads, -1)
        q_proj = rms_norm(q_proj, attn_layer.q_norm)
        q_proj = apply_rope(q_proj, segment_pos, cfg.rope_theta, cfg.rope_scaling)
        
    with jax.named_scope("k_proj"):
        k_proj = jnp.einsum("bld,dr->blr", x, attn_layer.k_proj)
        k_proj = k_proj.astype(dtype)
        k_proj = k_proj.reshape(k_proj.shape[0], k_proj.shape[1], cfg.num_key_value_heads, -1)
        k_proj = rms_norm(k_proj, attn_layer.k_norm)
        k_proj = apply_rope(k_proj, segment_pos, cfg.rope_theta, cfg.rope_scaling)
        
    with jax.named_scope("v_proj"):
        v_proj = jnp.einsum("bld,dr->blr", x, attn_layer.v_proj)
        v_proj = v_proj.astype(dtype)
        v_proj = v_proj.reshape(v_proj.shape[0], v_proj.shape[1], cfg.num_key_value_heads, -1)
    
    with jax.named_scope("cache_update"):
        if cache is not None:
            end_index = cache.end_index[idx][0]
            slice_indices = (0, end_index%cache.v_cache[idx].shape[1], 0, 0)
            k_proj = jax.lax.dynamic_update_slice(cache.k_cache[idx], k_proj, slice_indices)
            v_proj = jax.lax.dynamic_update_slice(cache.v_cache[idx], v_proj, slice_indices)
            cache_updates = (k_proj, v_proj)
        else:
            cache_updates = None
    
    with jax.named_scope("attention"):
        attn_out = attention(q_proj, 
                             k_proj, 
                             v_proj, 
                             segment_pos, 
                             cache.end_index[idx][0] if cache is not None else 0, 
                             attn_mask, 
                             cfg)
    
    with jax.named_scope("o_proj"):
        attn_out = jnp.einsum("bld,dD->blD", attn_out, attn_layer.o_proj)
    return attn_out, cache_updates
     
def mlp_block(
    x: Float[Array, "B L D"],
    layer: MLPLayer,
    cfg: Config
) -> Float[Array, "B L D"]:
    dtype = cfg.dtype
    with jax.named_scope("gate"):
        ff_gate = jax.nn.silu(jnp.einsum("btd,df->btf", x, layer.gate_proj)).astype(dtype)
    with jax.named_scope("up_proj"):
        ff_up = jnp.einsum("btd,df->btf", x, layer.up_proj).astype(dtype)
    with jax.named_scope("down_proj"):
        ff_out = jnp.einsum("btf, fd->btd", ff_gate * ff_up, layer.down_proj).astype(dtype)
    return ff_out

def forward_layer(
    x: Float[Array, "B L D"],
    segment_pos: Int[Array, "B L"],
    attn_mask: Bool[Array, "B L S"],
    layer: Layer,
    cfg: Config,
    idx: int,
    cache: KVCache | None = None,
) -> tuple[Float[Array, "B L D"], tuple]:
    x = x.astype(cfg.dtype)
    
    #Attention block
    with jax.named_scope("attn_pre_norm"):
        attn_in = rms_norm(x, layer.input_layernorm)
    attn_out, cache_updates = attention_block(
        x=attn_in, segment_pos=segment_pos, attn_mask=attn_mask, 
        attn_layer=layer.attention, cfg=cfg, cache=cache, idx=idx
    )
    with jax.named_scope("residual"):
        x = x + attn_out.astype(cfg.dtype)
    
    #FFN block
    with jax.named_scope("attn_post_norm"):
        ff_in = rms_norm(x, layer.post_attention_layernorm)
    with jax.named_scope("ffn"):
        ff_out = mlp_block(ff_in, layer.mlp, cfg)
    with jax.named_scope("residual"):
        x = x + ff_out.astype(cfg.dtype)
    
    return x, cache_updates

def forward(
    x: Int[Array, "B L"],
    segment_pos: Int[Array, "B L"],
    attn_mask: Bool[Array, "B L S"],
    weights: Weights,
    cfg: Config,
    cache: KVCache | None = None
):
    with jax.named_scope("vocab_in_proj"):
        # Embed input tokens [B, L] -> [B, L D]
        x = weights.embedding[x]
        
    for idx, layer in enumerate(weights.layers):
        x, cache_updates = forward_layer(x=x, segment_pos=segment_pos, attn_mask=attn_mask, 
                                         layer=layer, cfg=cfg, idx=idx, cache=cache)
        if cache is not None:
            cache.end_index[idx] += x.shape[1]
            cache.k_cache[idx], cache.v_cache[idx] = cache_updates
    x = rms_norm(x, weights.final_norm)
    with jax.named_scope("vocab_out_proj"):
        logits = jnp.einsum("btd,vd->btv", x, weights.embedding)
        
    return logits, cache

def create_reference_model(
    live_weights: Weights,
):
    """
    Returns an immutable, gradient-detached snapshot of live_params.
    
    * leaves are copied so later in-place edits won't alias
    * every leaf is wrapped in `lax.stop_gradient` so the backward
      pass ignores the reference copy
    * the resulting tree is marked immutable by converting lists -> tuples
      (purely defensive; JAX itself doesn't mutate in place).
    """
    def _freeze_leaf(x):
        x = jnp.array(x, copy=True)
        return jax.lax.stop_gradient(x)
    
    ref = tree_util.tree_map(_freeze_leaf, live_weights)
    
    def _list2tuple(x):
        return tuple(x) if isinstance(x, list) else x
    
    ref = tree_util.tree_map(_list2tuple, ref, is_leaf=lambda x: isinstance(x, jax.Array))
    
    return ref

def save_pytree(data, path):
    import orbax.checkpoint as ocp

    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(epath.Path(path), data, ocp.args.PyTreeSave(data))


def load_pytree(path, model_template = Weights):
    import orbax.checkpoint as ocp
    from orbax.checkpoint import RestoreArgs
    
    item, transforms = model_template, None
    # restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            epath.Path(path), args=ocp.args.PyTreeRestore(item=item, 
                                                          transforms=transforms)
        )