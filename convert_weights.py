from pathlib import Path
from argparse import ArgumentParser
import shutil

from qwen import model as qwenjax
import chkpt_utils as utils

from transformers import AutoConfig
from safetensors import safe_open
from tqdm import tqdm


def main(model_path: str | Path, ckpt_path: str | Path):
    model_path, ckpt_path = Path(model_path).expanduser(), Path(ckpt_path).expanduser()
    files = list(model_path.glob("**/*safetensors"))
    assert len(files) >= 1
    config_files = list(model_path.glob("**/config.json"))
    assert len(config_files) == 1, "Must have only one `config.json` file in the model path"
    config = AutoConfig.from_pretrained(config_files[0])
    cfg = qwenjax.hf_to_jax_config(config)
    weights = qwenjax.Weights.abstract(cfg)

    if not ckpt_path.exists():
        model = {}
        for file in tqdm(files):
            with safe_open(file, framework="torch") as f:
                for key in tqdm(f.keys(), leave=False):
                    model[key] = f.get_tensor(key)
        converted_weights = utils.convert_model_or_layer(weights, model, cfg, sequential=False, prefix="model.")
        qwenjax.save_pytree(converted_weights, ckpt_path)

    additional_files = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    for additional_file in additional_files:
        full_paths = list(model_path.glob(f"**/{additional_file}"))
        if len(full_paths) != 1:
            print(f"Found more than 1 file for {additional_file}")
        if len(full_paths) == 0:
            continue
        full_path = full_paths[0]
        shutil.copyfile(full_path, ckpt_path / full_path.name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source-path", default="~/Qwen3-0.6B-Base", required=True, help="HF model directory path"
    )
    parser.add_argument(
        "--dest-path",
        default="~/Qwen3-0.6B-Base-jax",
        required=True,
        help="JAX model model directory (to be created).",
    )
    args = parser.parse_args()
    main(args.source_path, args.dest_path)