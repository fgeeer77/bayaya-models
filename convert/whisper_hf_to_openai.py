#!/usr/bin/env python3
"""Converts a Hugging Face transformers Whisper checkpoint into the openai-whisper format, which the
official sherpa-onnx export script (scripts/whisper/export-onnx.py) takes as input.

    python3 whisper_hf_to_openai.py <hf repo> <revision> <output.pt>
"""
import glob
import json
import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

# transformers name fragment -> openai-whisper name fragment, applied in order.
RENAMES = [
    ("layers", "blocks"),
    (".self_attn.q_proj", ".attn.query"),
    (".self_attn.k_proj", ".attn.key"),
    (".self_attn.v_proj", ".attn.value"),
    (".self_attn_layer_norm", ".attn_ln"),
    (".self_attn.out_proj", ".attn.out"),
    (".encoder_attn.q_proj", ".cross_attn.query"),
    (".encoder_attn.k_proj", ".cross_attn.key"),
    (".encoder_attn.v_proj", ".cross_attn.value"),
    (".encoder_attn_layer_norm", ".cross_attn_ln"),
    (".encoder_attn.out_proj", ".cross_attn.out"),
    ("fc1", "mlp.0"),
    ("fc2", "mlp.2"),
    ("final_layer_norm", "mlp_ln"),
    ("decoder.layer_norm.", "decoder.ln."),
    ("encoder.layer_norm.", "encoder.ln_post."),
    ("embed_tokens", "token_embedding"),
    ("encoder.embed_positions.weight", "encoder.positional_embedding"),
    ("decoder.embed_positions.weight", "decoder.positional_embedding"),
]


def main():
    repo, revision, output = sys.argv[1:4]
    path = snapshot_download(repo, revision=revision, allow_patterns=["config.json", "*.safetensors", "*.safetensors.index.json"])
    config = json.load(open(f"{path}/config.json"))
    state = {}
    for f in sorted(glob.glob(f"{path}/*.safetensors")):
        state.update(load_file(f))
    if not state:
        raise SystemExit("no safetensors weights found")
    dims = {
        "n_mels": config["num_mel_bins"],
        "n_vocab": config["vocab_size"],
        "n_audio_ctx": config["max_source_positions"],
        "n_audio_state": config["d_model"],
        "n_audio_head": config["encoder_attention_heads"],
        "n_audio_layer": config["encoder_layers"],
        "n_text_ctx": config["max_target_positions"],
        "n_text_state": config["d_model"],
        "n_text_head": config["decoder_attention_heads"],
        "n_text_layer": config["decoder_layers"],
    }
    converted = {}
    for key, value in state.items():
        if key == "proj_out.weight":  # tied to the token embedding
            continue
        name = key.removeprefix("model.")
        for old, new in RENAMES:
            name = name.replace(old, new)
        converted[name] = value.to(torch.float16)  # halves the file; loaded back as float32
    torch.save({"dims": dims, "model_state_dict": converted}, output)
    print(f"{repo}@{revision}: {len(converted)} tensors, dims {dims} -> {output}")


if __name__ == "__main__":
    main()
