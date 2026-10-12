#!/usr/bin/env python3
"""Silero TTS v5 cis_base (MIT) for ONNX Runtime.

The original is one TorchScript model: text symbols and a speaker in, 48 kHz audio out. Its networks
are exported to ONNX one by one and the glue between them is left to the app (and to `synthesize`
below, the same steps in numpy, used to check the export against the original):

    tokens = [2] + symbol ids of the lower-case text + [1]
    durations.onnx (tokens, speaker) -> log_durations;  frames = round(exp(x) - 1), first <= 5, >= 0
    encoder.onnx   (tokens, speaker) -> encoded, pitch            (one vector per symbol)
    frames of every symbol repeated, encoded + pitch added
    decoder.onnx   (frames) -> magnitude, real, imag              (2400-point spectra, 600 apart)
    inverse STFT with "same" padding and a Hann window -> 48 kHz samples

Stress: Russian, Ukrainian and Belarusian words need a "+" before the stressed vowel.

    python3 tts_silero.py export <model.jit> <out dir>
    python3 tts_silero.py check <model.jit> <out dir>      # ONNX pipeline vs. the original
"""
import json
import os
import sys
from typing import Tuple

import numpy as np

SYMBOLS = "|||!'+,-.:;?hабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїјўґғҕҗҙқҝҡңҥҫүұҳҷҹһӑӗәӝӟӣӥӧөӯӱӳӵӏ—… "
SPEAKERS = [
    "aze_gamat", "bak_aigul", "bak_alfia", "bak_alfia2", "bak_miyau", "bak_ramilia", "bel_anatoliy", "bel_dmitriy",
    "bel_larisa", "chv_ekaterina", "erz_alexandr", "hye_zara", "kat_vika", "kaz_zhadyra", "kaz_zhazira", "kbd_eduard",
    "kir_nurgul", "kjh_karina", "kjh_sibday", "mdf_oksana", "ru_aigul", "ru_albina", "ru_alexandr", "ru_alfia",
    "ru_alfia2", "ru_bogdan", "ru_dmitriy", "ru_eduard", "ru_ekaterina", "ru_gamat", "ru_igor", "ru_karina",
    "ru_kejilgan", "ru_kermen", "ru_marat", "ru_miyau", "ru_nurgul", "ru_oksana", "ru_onaoy", "ru_ramilia", "ru_roman",
    "ru_safarhuja", "ru_saida", "ru_sibday", "ru_vika", "ru_zara", "ru_zhadyra", "ru_zhazira", "ru_zinaida",
    "sah_zinaida", "tat_albina", "tat_marat", "tgk_onaoy", "tgk_safarhuja", "udm_bogdan", "ukr_igor", "ukr_roman",
    "uzb_saida", "xal_kejilgan", "xal_kermen",
]
# Silero's language prefixes and the codes the app uses (ISO 639-1 where one exists).
LANGUAGES = {
    "aze": "az", "bak": "ba", "bel": "be", "chv": "cv", "erz": "myv", "hye": "hy", "kat": "ka", "kaz": "kk",
    "kbd": "kbd", "kir": "ky", "kjh": "kjh", "mdf": "mdf", "ru": "ru", "sah": "sah", "tat": "tt", "tgk": "tg",
    "udm": "udm", "ukr": "uk", "uzb": "uz", "xal": "xal",
}
N_FFT, HOP = 2400, 600


def tokens(text):
    ids = {s: i for i, s in enumerate(SYMBOLS)}
    return np.array([[2] + [ids[c] for c in text.lower() if c in ids] + [1]], dtype=np.int64)


def export(jit, out):
    import torch

    model = torch.jit.load(jit, map_location="cpu")
    model.eval()

    class Durations(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.durations = m.dur_predictor

        def forward(self, tokens: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
            return self.durations(tokens, speaker, None, 1.0, None)

    class Encoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.pitch = m.pitch_predictor
            self.encoder = m.mel_encoder

        def forward(self, tokens: torch.Tensor, speaker: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # The pitch predictor does not look at the durations (it passes None on).
            pitch = self.pitch(tokens, speaker, torch.zeros_like(tokens), 1.0, None)
            zero = torch.zeros(1, dtype=torch.long)
            return self.encoder(tokens, speaker, zero, zero, pitch, None)

    class Decoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.decoder = m.mel_decoder
            self.vocoder = m.vocoder

        def forward(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.vocoder(self.decoder(frames), 48000, 0.0, True)

    x = torch.from_numpy(tokens("пр+ивет, как дел+а?"))
    spk = torch.LongTensor([26])
    with torch.no_grad():
        enc, proj = Encoder(model)(x, spk)
        frames = torch.repeat_interleave(enc[0] + proj[0], 5, 0)[None]
    os.makedirs(out, exist_ok=True)
    parts = [
        ("durations", Durations(model), (x, spk), ["tokens", "speaker"], ["log_durations"],
         {"tokens": {1: "symbols"}, "log_durations": {1: "symbols"}}),
        ("encoder", Encoder(model), (x, spk), ["tokens", "speaker"], ["encoded", "pitch"],
         {"tokens": {1: "symbols"}, "encoded": {1: "symbols"}, "pitch": {1: "symbols"}}),
        ("decoder", Decoder(model), (frames,), ["frames"], ["magnitude", "real", "imag"],
         {"frames": {1: "frames"}, "magnitude": {2: "spectra"}, "real": {2: "spectra"}, "imag": {2: "spectra"}}),
    ]
    for name, module, args, inputs, outputs, axes in parts:
        module.eval()
        path = os.path.join(out, f"{name}.onnx")
        # The networks are TorchScript: tracing a Python wrapper cannot step into them, so the wrapper
        # itself is scripted (and, if the exporter still refuses, frozen into one inlined graph).
        errors = []
        for how in ["scripted", "frozen"]:
            try:
                scripted = torch.jit.script(module)
                if how == "frozen":
                    scripted = torch.jit.freeze(scripted.eval())
                torch.onnx.export(scripted, args, path, opset_version=17, input_names=inputs, output_names=outputs,
                                  dynamic_axes=axes, do_constant_folding=True)
                print(name, how, os.path.getsize(path), "bytes")
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{how}: {type(e).__name__}: {str(e)[:1500]}")
        else:
            raise SystemExit(f"{name} could not be exported:\n" + "\n".join(errors))
    meta = {"symbols": SYMBOLS, "bos": 2, "eos": 1, "sample_rate": 48000, "n_fft": N_FFT, "hop": HOP,
            "speakers": [f"{LANGUAGES[s.split('_')[0]]}:{s}" for s in SPEAKERS]}
    json.dump(meta, open(os.path.join(out, "silero.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def istft(magnitude, real, imag):
    """Vocos ISTFT, padding "same": magnitude/real/imag are (bins, frames)."""
    spec = magnitude * (real + 1j * imag)
    frames = spec.shape[1]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    total = (frames - 1) * HOP + N_FFT
    y = np.zeros(total)
    envelope = np.zeros(total)
    chunks = np.fft.irfft(spec, N_FFT, axis=0) * window[:, None]
    for t in range(frames):
        y[t * HOP:t * HOP + N_FFT] += chunks[:, t]
        envelope[t * HOP:t * HOP + N_FFT] += window ** 2
    pad = (N_FFT - HOP) // 2
    return (y[pad:-pad] / envelope[pad:-pad]).astype(np.float32)


class Synthesizer:
    """The app's pipeline in numpy and ONNX Runtime."""

    def __init__(self, directory, threads=4):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.durations = ort.InferenceSession(os.path.join(directory, "durations.onnx"), options)
        self.encoder = ort.InferenceSession(os.path.join(directory, "encoder.onnx"), options)
        self.decoder = ort.InferenceSession(os.path.join(directory, "decoder.onnx"), options)

    def __call__(self, text, speaker):
        x = tokens(text)
        spk = np.array([speaker], dtype=np.int64)
        log = self.durations.run(None, {"tokens": x, "speaker": spk})[0][0]
        frames = np.round(np.exp(log) - 1).astype(np.int64)
        frames[0] = min(frames[0], 5)
        frames[frames < 0] = 0
        encoded, pitch = self.encoder.run(None, {"tokens": x, "speaker": spk})
        expanded = np.repeat(encoded[0] + pitch[0], frames, axis=0)[None].astype(np.float32)
        magnitude, real, imag = self.decoder.run(None, {"frames": expanded})
        return istft(magnitude[0], real[0], imag[0])


def check(jit, directory):
    import torch

    model = torch.jit.load(jit, map_location="cpu")
    model.eval()
    synthesizer = Synthesizer(directory)
    worst = 1e9
    for text, speaker in [("в н+едрах т+ундры в+ыдры в г+етрах т+ырят в в+ёдра +ядра к+едров.", 26),
                          ("Мин китап уҡырға яратам.", 2), ("+я з р+аннього дит+инства д+уже любл+ю к+азки.", 56)]:
        with torch.no_grad():
            want = model(torch.from_numpy(tokens(text)), torch.LongTensor([speaker]))[0].numpy()
        got = synthesizer(text, speaker)
        n = min(len(want), len(got))
        noise = np.sum((want[:n] - got[:n]) ** 2) + 1e-12
        snr = 10 * np.log10(np.sum(want[:n] ** 2) / noise)
        worst = min(worst, snr)
        print(f"speaker {speaker}: original {len(want)} samples, ONNX {len(got)}, SNR {snr:.1f} dB")
    if worst < 40:
        raise SystemExit(f"the ONNX pipeline differs from the original (SNR {worst:.1f} dB)")


if __name__ == "__main__":
    {"export": lambda: export(sys.argv[2], sys.argv[3]), "check": lambda: check(sys.argv[2], sys.argv[3])}[sys.argv[1]]()
