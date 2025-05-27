import re
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import torch
from tqdm import tqdm

from qwen import model as qwenjax

is_leaf = lambda x: isinstance(x, qwenjax.ArrayInfo)

def t2j(x):
    return jnp.from_dlpack(x.detach().contiguous())

def _index_to_str(x):
    """Convert objects from jax.tree.flatten_with_path to dot separated strings."""
    for field in ["name", "idx", "key"]:
        if hasattr(x, field):
            return str(getattr(x, field))
    raise ValueError

def convert_weights(key: str, value: torch.Tensor, cfg: qwenjax.Config):
    value = value.detach()
    # HF checkpoint naming convention ------------------------------------------
    # attention ################################################################
    if re.search(r"q_proj\.weight", key) is not None:
        assert value.shape == (cfg.num_attention_heads*cfg.head_dim, cfg.hidden_size)
        return t2j(value.T)
    elif re.search(r"k_proj\.weight", key) is not None:
        assert value.shape == (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
        return t2j(value.T)
    elif re.search(r"v_proj\.weight", key) is not None:
        assert value.shape == (cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size)
        return t2j(value.T)
    elif re.search(r"o_proj\.weight", key) is not None:
        assert value.shape == (cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim)
        return t2j(value.T)
    elif re.search(r"q_norm\.weight", key) is not None:
        assert value.shape == (cfg.head_dim,)
        return t2j(value)
    elif re.search(r"k_norm\.weight", key) is not None:
        assert value.shape == (cfg.head_dim,)
        return t2j(value)
    # MLP ######################################################################
    elif re.search(r"gate_proj\.weight", key) is not None:
        assert value.shape == (cfg.intermediate_size, cfg.hidden_size)
        return t2j(value.T)
    elif re.search(r"up_proj\.weight", key) is not None:
        assert value.shape == (cfg.intermediate_size, cfg.hidden_size)
        return t2j(value.T)
    elif re.search(r"down_proj\.weight", key) is not None:
        assert value.shape == (cfg.hidden_size, cfg.intermediate_size)
        return t2j(value.T)
    # shared misc weights ------------------------------------------------------
    # misc #####################################################################
    elif re.search(r"embed_tokens", key) is not None:
        assert value.shape == (cfg.vocab_size, cfg.hidden_size)
        return t2j(value)
    elif re.search(r"input_layernorm\.weight", key) is not None:
        assert value.shape == (cfg.hidden_size,)
        return t2j(value)
    elif re.search(r"post_attention_layernorm\.weight", key) is not None:
        assert value.shape == (cfg.hidden_size,)
        return t2j(value)
    elif re.search(r"model\.norm\.weight", key) is not None:
        assert value.shape == (cfg.hidden_size,)
        return t2j(value)
    else:
        raise ValueError(f"Unknown weight key {key = }")
    
_HF_KEY_MAPPING = {
    # Embedding
    r"model\.embed_tokens\.weight": "embedding",
    
    # Attention Layers
    r"model\.layers\.(\d+)\.self_attn\.q_proj\.weight": r"layers.\1.attention.q_proj",
    r"model\.layers\.(\d+)\.self_attn\.k_proj\.weight": r"layers.\1.attention.k_proj",
    r"model\.layers\.(\d+)\.self_attn\.v_proj\.weight": r"layers.\1.attention.v_proj",
    r"model\.layers\.(\d+)\.self_attn\.o_proj\.weight": r"layers.\1.attention.o_proj",
    r"model\.layers\.(\d+)\.self_attn\.q_norm\.weight": r"layers.\1.attention.q_norm",
    r"model\.layers\.(\d+)\.self_attn\.k_norm\.weight": r"layers.\1.attention.k_norm",
    
    # MLP Layers
    r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight": r"layers.\1.mlp.gate_proj",
    r"model\.layers\.(\d+)\.mlp\.up_proj\.weight": r"layers.\1.mlp.up_proj",
    r"model\.layers\.(\d+)\.mlp\.down_proj\.weight": r"layers.\1.mlp.down_proj",
    
    # Layer Norm
    r"model\.layers\.(\d+)\.input_layernorm\.weight": r"layers.\1.input_layernorm",
    r"model\.layers\.(\d+)\.post_attention_layernorm\.weight": r"layers.\1.post_attention_layernorm",
    
    # Final Layer Norm
    r"model\.norm\.weight": r"final_norm",
}

def _qwen_key_to_jax_key(qwen_key):
    for pat, repl in _HF_KEY_MAPPING.items():
        m = re.match(pat, qwen_key)
        if m is None:
            continue
        return re.sub(pat, repl, qwen_key)
    return None

def convert_model_or_layer(
    layer: qwenjax.Weights | qwenjax.Layer,
    qwen_layer: torch.nn.Module,
    cfg: qwenjax.Config,
    device: jax.Device | None = None,
    sequential: bool = False,
    allow_unconverted_parameters: bool = False,
    prefix: str | None = None,
):
    device = device if device is not None else jax.devices("cpu")[0]
    torch_params = dict(qwen_layer.named_parameters() if hasattr(qwen_layer, "named_parameters") else qwen_layer)
    torch_params = {k: v for (k,v) in torch_params.items() if prefix is None or k.startswith(prefix)}
    
    layer_params = {
        ".".join(map(_index_to_str, k)): v for (k, v) in jax.tree.flatten_with_path(layer, is_leaf=is_leaf)[0]
    }
    new_params = {k: None for k in layer_params.keys()}
    
    def convert_weight_thread(tkey, tweight):
        with jax.default_device(device):
            jweight = convert_weights(tkey, tweight, cfg)
        jkey = _qwen_key_to_jax_key(tkey)
        if jkey is None:
            raise ValueError(f"Could not find parameter mapping for torch paramter: `{tkey}`.")
        if jkey not in new_params:
            raise ValueError(f"The JAX model is not expecting `{jkey}`!  Expected keys are {list(new_params.keys())}")
        if new_params[jkey] is not None:
            raise ValueError(f"Parameter `{jkey}` already set!")
        new_params[jkey] = jweight

    if sequential:
        for tkey, tweight in torch_params.items():
            convert_weight_thread(tkey, tweight)
    else:
        futures, executor = [], ThreadPoolExecutor(max_workers=16)
        for tkey, tweight in torch_params.items():
            futures.append(executor.submit(convert_weight_thread, tkey, tweight))
        for fut in tqdm(futures, desc="Converting weights"):
            fut.result()

    if not allow_unconverted_parameters:
        assert all(v is not None for v in new_params.values()), str({k: v for k, v in new_params.items() if v is None})

    if isinstance(layer, qwenjax.Weights):
        return jax.tree.unflatten(jax.tree.structure(layer, is_leaf=is_leaf), new_params.values())
    else:
        return jax.tree.unflatten(
            jax.tree.structure(layer, is_leaf=is_leaf),
            [
                new_param if new_param is not None else param
                for (new_param, param) in zip(new_params.values(), layer_params.values())
            ],
        )