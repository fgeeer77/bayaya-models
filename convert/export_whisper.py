#!/usr/bin/env python3
"""Runs the official sherpa-onnx Whisper export (pinned release) on a converted checkpoint.

The upstream script only knows OpenAI's model names; three small, asserted patches let it load
./<name>.pt instead and take the mel size from the checkpoint.

    python3 export_whisper.py <name>     # expects ./<name>.pt, writes <name>-{encoder,decoder}.int8.onnx, <name>-tokens.txt
"""
import runpy
import sys
import urllib.request

UPSTREAM = "https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/v1.13.8/scripts/whisper/export-onnx.py"


def patch(src: str, old: str, new: str) -> str:
    if src.count(old) != 1:
        raise SystemExit(f"upstream script changed, cannot patch: {old!r}")
    return src.replace(old, new)


def main():
    name = sys.argv[1]
    src = urllib.request.urlopen(UPSTREAM, timeout=60).read().decode()
    src = patch(src, '            "medium-aishell",\n', f'            "medium-aishell",\n            "{name}",\n')
    src = patch(src, "def load_model(name: str):\n",
                "def load_model(name: str):\n    if os.path.isfile(f\"./{name}.pt\"):\n        return whisper.load_model(f\"./{name}.pt\")\n")
    src = patch(src, "    mel = (\n        whisper.log_mel_spectrogram", "    n_mels = model.dims.n_mels\n    mel = (\n        whisper.log_mel_spectrogram")
    with open("sherpa_export_whisper.py", "w") as f:
        f.write(src)
    sys.argv = ["sherpa_export_whisper.py", "--model", name]
    runpy.run_path("sherpa_export_whisper.py", run_name="__main__")


if __name__ == "__main__":
    main()
