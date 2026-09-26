#!/usr/bin/env python3
"""Transcribes the test recording with the ORIGINAL model (transformers / gigaam) for verify.py.

    python3 reference.py whisper <hf repo> <revision> <wav> <out.txt>
    python3 reference.py gigaam <model name> <wav> <out.txt>
"""
import sys


def main():
    kind = sys.argv[1]
    if kind == "whisper":
        repo, revision, wav, out = sys.argv[2:6]
        import soundfile as sf
        import torch
        from transformers import pipeline

        audio, rate = sf.read(wav, dtype="float32")
        asr = pipeline("automatic-speech-recognition", model=repo, revision=revision, torch_dtype=torch.float32, device="cpu")
        text = asr({"raw": audio, "sampling_rate": rate}, generate_kwargs={"language": "russian", "task": "transcribe"})["text"]
    else:
        name, wav, out = sys.argv[2:5]
        import gigaam

        text = gigaam.load_model(name).transcribe(wav)
        text = getattr(text, "text", text)
    open(out, "w", encoding="utf-8").write(str(text).strip())
    print(text)


if __name__ == "__main__":
    main()
