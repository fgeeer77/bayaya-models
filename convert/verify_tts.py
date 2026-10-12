#!/usr/bin/env python3
"""Checks a converted voice the way a listener would: sherpa-onnx (the runtime the app uses) reads
known sentences with every speaker, a speech recogniser writes down what it hears, and the job fails
if that differs from the text by more than the threshold (character error rate, averaged).

    python3 verify_tts.py kokoro <model dir> <model file> <espeak voice> <asr: gigaam|whisper> <asr dir>
        <sentences file> <max cer> [<out dir for the wav files>]
"""
import os
import sys
import time

import sherpa_onnx
import soundfile as sf

from verify import cer


def synthesizer(kind: str, directory: str, model: str, voice: str, espeak: str):
    if kind == "kokoro":
        config = sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=os.path.join(directory, model),
                voices=os.path.join(directory, "voices.bin"),
                tokens=os.path.join(directory, "tokens.txt"),
                data_dir=espeak,
                lang=voice,
            ),
            num_threads=4,
        )
    else:
        raise SystemExit(f"unknown kind {kind}")
    return sherpa_onnx.OfflineTts(sherpa_onnx.OfflineTtsConfig(model=config))


def recognizer(kind: str, directory: str, language: str):
    if kind == "gigaam":
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=os.path.join(directory, "encoder.int8.onnx"),
            decoder=os.path.join(directory, "decoder.onnx"),
            joiner=os.path.join(directory, "joiner.onnx"),
            tokens=os.path.join(directory, "tokens.txt"),
            model_type="nemo_transducer",
            num_threads=4,
        )
    name = os.path.basename(directory.rstrip("/")).removeprefix("sherpa-onnx-whisper-")
    return sherpa_onnx.OfflineRecognizer.from_whisper(
        encoder=os.path.join(directory, f"{name}-encoder.int8.onnx"),
        decoder=os.path.join(directory, f"{name}-decoder.int8.onnx"),
        tokens=os.path.join(directory, f"{name}-tokens.txt"),
        language=language,
        task="transcribe",
        num_threads=4,
    )


def main():
    kind, directory, model, voice, asr_kind, asr_dir, sentences, max_cer = sys.argv[1:9]
    out = sys.argv[9] if len(sys.argv) > 9 else None
    espeak = os.environ.get("ESPEAK_DATA", os.path.join(directory, "espeak-ng-data"))
    texts = [line.strip() for line in open(sentences, encoding="utf-8") if line.strip()]
    tts = synthesizer(kind, directory, model, voice, espeak)
    asr = recognizer(asr_kind, asr_dir, voice.split("-")[0])
    scores = []
    work = audio = 0.0
    for sid in range(tts.num_speakers):
        for i, text in enumerate(texts):
            config = sherpa_onnx.GenerationConfig()
            config.sid = sid
            started = time.time()
            generated = tts.generate(text, config)
            work += time.time() - started
            audio += len(generated.samples) / generated.sample_rate
            if out:
                os.makedirs(out, exist_ok=True)
                sf.write(os.path.join(out, f"{sid}-{i}.wav"), generated.samples, generated.sample_rate)
            stream = asr.create_stream()
            stream.accept_waveform(generated.sample_rate, generated.samples)
            asr.decode_stream(stream)
            heard = stream.result.text
            score = cer(text, heard)
            scores.append(score)
            print(f"speaker {sid}: CER {score:.3f}\n  text:  {text}\n  heard: {heard}")
    mean = sum(scores) / len(scores)
    print(f"{model}: mean CER {mean:.3f} (max {max_cer}), {tts.num_speakers} speakers, "
          f"real-time factor {work / max(audio, 1e-3):.2f} on the runner")
    if mean > float(max_cer):
        raise SystemExit("the converted voice is not understood well enough")


if __name__ == "__main__":
    main()
