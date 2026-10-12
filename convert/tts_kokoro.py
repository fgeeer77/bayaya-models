#!/usr/bin/env python3
"""Turns a Kokoro-family ONNX export into a sherpa-onnx Kokoro model.

The source is a release of github.com/buchwandler/kokoro-onnx-models (checkpoints converted to ONNX
with the same inputs sherpa-onnx feeds Kokoro: tokens, style, speed). This adds the metadata sherpa-onnx
reads, writes a dynamically quantised int8 copy next to the full one, and takes tokens.txt from the
official sherpa-onnx Kokoro 1.0 (the vocabulary is the same; checked). eSpeak NG turns the text into
phonemes; its data is not copied: the app already has the same directory, pinned by its digest.

    python3 tts_kokoro.py <out dir> <kokoro-1.0 dir> <model url> <model sha256> <voices url> <voices sha256>
        <config url> <config sha256> <speakers, comma separated> <espeak voice> <language name>
"""
import hashlib
import json
import os
import shutil
import sys
import urllib.request

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def download(url: str, sha256: str, path: str):
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as response, open(path, "wb") as out:
        while chunk := response.read(1 << 20):
            digest.update(chunk)
            out.write(chunk)
    if digest.hexdigest() != sha256:
        raise SystemExit(f"{url}: sha256 {digest.hexdigest()} != {sha256}")


def main():
    out, reference, model_url, model_sha, voices_url, voices_sha, config_url, config_sha, speakers, voice, language = sys.argv[1:12]
    names = speakers.split(",")
    os.makedirs(out, exist_ok=True)
    source = os.path.join(out, "source.onnx")
    download(model_url, model_sha, source)
    download(voices_url, voices_sha, os.path.join(out, "voices.bin"))
    download(config_url, config_sha, os.path.join(out, "source-config.json"))

    # tokens.txt comes from Kokoro 1.0: the model must have been trained with the same vocabulary.
    vocab = json.load(open(os.path.join(out, "source-config.json"), encoding="utf-8"))["vocab"]
    tokens = {}
    for line in open(os.path.join(reference, "tokens.txt"), encoding="utf-8"):
        parts = line.rstrip("\n").rsplit(" ", 1)
        tokens[parts[0] if parts[0] else " "] = int(parts[1])
    if vocab != tokens:
        raise SystemExit("the model's vocabulary differs from Kokoro 1.0's tokens.txt")
    os.remove(os.path.join(out, "source-config.json"))

    # One style table of 510 x 256 floats per speaker.
    size = os.path.getsize(os.path.join(out, "voices.bin"))
    if size != len(names) * 510 * 256 * 4:
        raise SystemExit(f"voices.bin holds {size} bytes, expected {len(names)} speakers")

    model = onnx.load(source)
    inputs = [i.name for i in model.graph.input]
    if inputs != ["tokens", "style", "speed"]:
        raise SystemExit(f"unexpected inputs {inputs}")
    meta = {
        "model_type": "kokoro",
        "language": language,
        "has_espeak": 1,
        "sample_rate": 24000,
        "version": 2,
        "voice": voice,
        "style_dim": "510,1,256",
        "n_speakers": len(names),
        "speaker_names": ",".join(names),
        "maintainer": "github.com/fgeeer77/bayaya-models",
    }
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.save(model, os.path.join(out, "model.onnx"))
    os.remove(source)
    quantize_dynamic(os.path.join(out, "model.onnx"), os.path.join(out, "model.int8.onnx"), weight_type=QuantType.QUInt8)

    shutil.copy(os.path.join(reference, "tokens.txt"), os.path.join(out, "tokens.txt"))
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
