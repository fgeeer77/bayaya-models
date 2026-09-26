#!/usr/bin/env python3
"""Checks a converted model with sherpa-onnx itself (the runtime the app uses) against the output
of the original model on the same recording. Fails if they differ by more than the threshold.

    python3 verify.py whisper <encoder> <decoder> <tokens> <wav> <reference.txt> [max_cer]
    python3 verify.py ctc <model> <tokens> <wav> <reference.txt> [max_cer]
"""
import re
import sys

import sherpa_onnx
import soundfile as sf


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def cer(ref: str, hyp: str) -> float:
    ref, hyp = normalize(ref), normalize(hyp)
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1] / max(1, len(ref))


def main():
    kind = sys.argv[1]
    if kind == "whisper":
        encoder, decoder, tokens, wav, reference = sys.argv[2:7]
        rest = sys.argv[7:]
        recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=encoder, decoder=decoder, tokens=tokens, language="ru", task="transcribe", num_threads=4)
    else:
        model, tokens, wav, reference = sys.argv[2:6]
        rest = sys.argv[6:]
        recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(model=model, tokens=tokens, num_threads=4)
    max_cer = float(rest[0]) if rest else 0.1
    samples, rate = sf.read(wav, dtype="float32", always_2d=True)
    stream = recognizer.create_stream()
    stream.accept_waveform(rate, samples[:, 0])
    recognizer.decode_stream(stream)
    text = stream.result.text
    expected = open(reference, encoding="utf-8").read()
    score = cer(expected, text)
    print(f"converted: {text}\noriginal:  {expected}\nCER {score:.3f} (max {max_cer})")
    if score > max_cer:
        raise SystemExit("converted model disagrees with the original")


if __name__ == "__main__":
    main()
