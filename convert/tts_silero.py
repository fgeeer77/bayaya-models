#!/usr/bin/env python3
"""Silero TTS v5 cis_base (MIT) for ONNX Runtime.

The original is one TorchScript model: text symbols and a speaker in, 48 kHz audio out. Its scripted
networks do not export (fused attention, mixed-precision checks), so they are rebuilt here as plain
PyTorch modules with the original's weights, checked part by part against it, and exported to ONNX.
The glue between them is left to the app (and to `Synthesizer` below, the same steps in numpy):

    tokens = [2] + symbol ids of the lower-case text + [1]
    encoder.onnx (tokens, speaker) -> log_durations, encoded      (one 128-vector per symbol)
    frames = round(exp(log_durations) - 1), the first at most 5, none negative
    every symbol's vector repeated for its frames
    decoder.onnx (frames) -> real, imag                           (2400-point spectra, 600 samples apart)
    inverse STFT with "same" padding and a Hann window -> 48 kHz samples

Stress: Russian, Ukrainian and Belarusian words need a "+" before the stressed vowel.

    python3 tts_silero.py export <model.jit> <out dir>
    python3 tts_silero.py check <model.jit> <out dir>      # ONNX pipeline vs. the original
"""
import json
import math
import os
import sys

import numpy as np
import torch

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


class Attention(torch.nn.Module):
    """nn.MultiheadAttention (batch first, no masks, one text at a time) written out: its fused fast path
    does not export."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.size = dim // heads
        self.scale = 1 / math.sqrt(self.size)
        self.in_proj_weight = torch.nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = torch.nn.Parameter(torch.empty(3 * dim))
        self.out_proj = torch.nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = torch.nn.functional.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q, k, v = (y.reshape(1, -1, self.heads, self.size).transpose(1, 2) for y in (q, k, v))
        weights = torch.softmax((q * self.scale) @ k.transpose(-1, -2), dim=-1)
        return self.out_proj((weights @ v).transpose(1, 2).reshape(1, -1, self.heads * self.size))


class FFTBlock(torch.nn.Module):
    """Attention and a convolution, each with a residual and LayerNorm (FastPitch's FFT block)."""

    def __init__(self, original):
        super().__init__()
        attention = original.self_attn
        dim = attention.in_proj_weight.shape[1]
        self.self_attn = Attention(dim, attention.num_heads)
        c1, c2 = original.conv1.weight.shape, original.conv2.weight.shape
        self.conv1 = torch.nn.Conv1d(c1[1], c1[0], c1[2], padding=c1[2] // 2)
        self.conv2 = torch.nn.Conv1d(c2[1], c2[0], c2[2], padding=c2[2] // 2)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.use_linear = bool(original.use_linear)
        self.pitch_pred = bool(original.pitch_pred)
        if self.use_linear:
            raise SystemExit("FFTBlock with use_linear: not seen in cis_base, not rebuilt")

    def forward(self, x):
        x = self.norm1(x + self.self_attn(x))
        h = x.transpose(1, 2)
        y = self.conv2(torch.relu(self.conv1(h)))
        if self.pitch_pred:
            # As the original: the activated output is added twice.
            y = torch.relu(y)
            h = h + y + y
        else:
            h = h + y
        return self.norm2(h.transpose(1, 2))


class PositionalEncoding(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.empty(1))
        self.register_buffer("pe", torch.empty(original.pe.shape))

    def forward(self, x):
        return x + self.scale * self.pe[: x.shape[1], 0].unsqueeze(0)


class ForwardTransformer(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        self.pos_encoder = PositionalEncoding(original.pos_encoder)
        self.layers = torch.nn.ModuleList(FFTBlock(layer) for layer in children(original.layers))
        self.norm = torch.nn.LayerNorm(original.norm.weight.shape[0])

    def forward(self, x):
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def children(original):
    """A scripted ModuleList's modules in order (it cannot be iterated itself)."""
    return [m for _, m in sorted(original.named_children(), key=lambda item: int(item[0]))]


def embedding(original):
    return torch.nn.Embedding(*original.weight.shape)


class SeriesPredictor(torch.nn.Module):
    """One value per symbol (log duration or pitch) for a speaker."""

    def __init__(self, original):
        super().__init__()
        self.embedding = embedding(original.embedding)
        self.speaker_embedding = embedding(original.speaker_embedding)
        self.emotion_embedding = embedding(original.emotion_embedding)
        self.transformer = ForwardTransformer(original.transformer)
        self.lin = torch.nn.Linear(original.lin.weight.shape[1], original.lin.weight.shape[0])

    def forward(self, tokens, speaker):
        x = self.embedding(tokens) + self.speaker_embedding(speaker).unsqueeze(1)
        return self.lin(self.transformer(x))


class Encoder(torch.nn.Module):
    """(tokens, speaker) -> log durations and one 128-vector per symbol (pitch already added)."""

    def __init__(self, original):
        super().__init__()
        encoder = original.mel_encoder
        if bool(encoder.articulation_emb):
            raise SystemExit("articulation embedding on: not seen in cis_base, not rebuilt")
        self.durations = SeriesPredictor(original.dur_predictor.dur_pred)
        self.pitch = SeriesPredictor(original.pitch_predictor.pitch_pred)
        self.embedding = embedding(encoder.embedding)
        self.speaker_embedding = embedding(encoder.speaker_embedding)
        self.encoder = ForwardTransformer(encoder.encoder)
        w = encoder.pitch_proj.weight.shape
        self.pitch_proj = torch.nn.Conv1d(w[1], w[0], w[2], padding=w[2] // 2)
        self.pitch_strength = float(encoder.pitch_strength)

    def forward(self, tokens, speaker):
        log_durations = self.durations(tokens, speaker).squeeze(-1)
        pitch = self.pitch(tokens, speaker).transpose(1, 2)
        encoded = self.encoder(self.embedding(tokens)) + self.speaker_embedding(speaker).unsqueeze(1)
        projected = self.pitch_proj(pitch).transpose(1, 2) * self.pitch_strength
        return log_durations, encoded + projected


class HourGlass(torch.nn.Module):
    """A layer at full rate, layers at a third of it, a layer at full rate again."""

    def __init__(self, original):
        super().__init__()
        self.pos_encoder = PositionalEncoding(original.pos_encoder)
        self.pre_vanilla_layers = torch.nn.ModuleList(FFTBlock(b) for b in children(original.pre_vanilla_layers))
        self.shorten_layers = torch.nn.ModuleList(FFTBlock(b) for b in children(original.shorten_layers))
        self.post_vanilla_layers = torch.nn.ModuleList(FFTBlock(b) for b in children(original.post_vanilla_layers))
        self.norm = torch.nn.LayerNorm(original.norm.weight.shape[0])
        self.factor = int(original.shorten_factor)
        self.dim = int(original.upsample.dim)
        w = original.upsample.proj.weight.shape
        self.upsample = torch.nn.Module()
        self.upsample.proj = torch.nn.Linear(w[1], w[0])
        if len(self.pre_vanilla_layers) != 1 or len(self.post_vanilla_layers) != 1:
            raise SystemExit("unexpected hourglass layout")

    def forward(self, x):
        length = x.shape[1]
        x = self.pre_vanilla_layers[0](self.pos_encoder(x))
        # Padding to a multiple of the factor, then average pooling: the same as appending factor - 1
        # zero frames and pooling without the incomplete last window.
        padded = torch.cat([x, torch.zeros_like(x[:, : self.factor - 1])], dim=1)
        short = torch.nn.functional.avg_pool1d(padded.transpose(1, 2), self.factor, self.factor).transpose(1, 2)
        # The original runs every shortened layer on the pooled frames but keeps only the last one's output.
        short = self.shorten_layers[-1](short)
        up = self.upsample.proj(short).reshape(x.shape[0], -1, self.dim)
        # Only the first `length` frames are kept, where the padded residual is x itself.
        x = up[:, :length] + x
        return self.norm(self.post_vanilla_layers[0](x))


class ConvNeXt(torch.nn.Module):
    def __init__(self, original):
        super().__init__()
        c = original.dwconv.weight.shape[0]
        k = original.dwconv.weight.shape[2]
        self.dwconv = torch.nn.Conv1d(c, c, k, padding=k // 2, groups=c)
        self.norm = torch.nn.LayerNorm(c, eps=1e-6)
        self.pwconv1 = torch.nn.Linear(c, original.pwconv1.weight.shape[0])
        self.pwconv2 = torch.nn.Linear(original.pwconv1.weight.shape[0], c)
        self.gamma = torch.nn.Parameter(torch.empty(c))

    def forward(self, x):
        y = self.norm(self.dwconv(x).transpose(1, 2))
        y = self.gamma * self.pwconv2(torch.nn.functional.gelu(self.pwconv1(y)))
        return x + y.transpose(1, 2)


class Decoder(torch.nn.Module):
    """Frames (one 128-vector per 600 samples) -> mel -> Vocos spectra (real and imaginary parts)."""

    def __init__(self, original):
        super().__init__()
        self.decoder = HourGlass(original.mel_decoder.decoder)
        w = original.mel_decoder.lin.weight.shape
        self.lin = torch.nn.Linear(w[1], w[0])
        backbone = original.vocoder.backbone
        w = backbone.embed.weight.shape
        self.embed = torch.nn.Conv1d(w[1], w[0], w[2], padding=w[2] // 2)
        self.norm = torch.nn.LayerNorm(w[0], eps=1e-6)
        self.convnext = torch.nn.ModuleList(ConvNeXt(b) for b in children(backbone.convnext))
        self.final_layer_norm = torch.nn.LayerNorm(w[0], eps=1e-6)
        w = original.vocoder.head.out.weight.shape
        self.out = torch.nn.Linear(w[1], w[0])

    def forward(self, frames):
        mel = self.lin(self.decoder(frames)).transpose(1, 2)
        x = self.norm(self.embed(mel).transpose(1, 2)).transpose(1, 2)
        for block in self.convnext:
            x = block(x)
        x = self.out(self.final_layer_norm(x.transpose(1, 2))).transpose(1, 2)
        log_magnitude, phase = x.chunk(2, dim=1)
        magnitude = torch.clamp(torch.exp(log_magnitude), max=100.0)
        return magnitude * torch.cos(phase), magnitude * torch.sin(phase)


# Where each rebuilt module's weights are in the original.
SOURCES = {
    "durations": "dur_predictor.dur_pred",
    "pitch": "pitch_predictor.pitch_pred",
    "embedding": "mel_encoder.embedding",
    "speaker_embedding": "mel_encoder.speaker_embedding",
    "encoder": "mel_encoder.encoder",
    "pitch_proj": "mel_encoder.pitch_proj",
    "decoder": "mel_decoder.decoder",
    "lin": "mel_decoder.lin",
    "embed": "vocoder.backbone.embed",
    "norm": "vocoder.backbone.norm",
    "convnext": "vocoder.backbone.convnext",
    "final_layer_norm": "vocoder.backbone.final_layer_norm",
    "out": "vocoder.head.out",
}


def load_weights(module, model):
    """Copies the original's parameters and buffers into the rebuilt module, checking every name."""
    original = {}
    for name, value in list(model.named_parameters()) + list(model.named_buffers()):
        original[name] = value.detach()
    state = {}
    for name in module.state_dict():
        head, _, rest = name.partition(".")
        source = SOURCES[head] + ("." + rest if rest else "")
        if source not in original:
            raise SystemExit(f"{name}: no {source} in the original")
        state[name] = original[source]
    module.load_state_dict(state, strict=True)
    return module.eval()


def load(jit):
    model = torch.jit.load(jit, map_location="cpu")
    model.eval()
    return model


def compare(name, got, want):
    diff = (got - want).abs().max().item()
    scale = want.abs().max().item()
    print(f"{name}: max difference {diff:.2e} (values up to {scale:.2e})")
    if diff > 1e-3 * max(scale, 1.0):
        raise SystemExit(f"{name}: the rebuilt network differs from the original")


def export(jit, out):
    model = load(jit)
    encoder = load_weights(Encoder(model), model)
    decoder = load_weights(Decoder(model), model)
    print("blocks with the doubled pitch residual:",
          [name for name, m in list(encoder.named_modules()) + list(decoder.named_modules())
           if isinstance(m, FFTBlock) and m.pitch_pred])

    # The rebuilt networks against the original's own parts, before anything is exported.
    x = torch.from_numpy(tokens("пр+ивет, как дел+а? в н+едрах т+ундры в+ыдры."))
    spk = torch.LongTensor([26])
    zero = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        log_durations, encoded = encoder(x, spk)
        compare("durations", log_durations, model.dur_predictor(x, spk, None, 1.0, None))
        pitch = model.pitch_predictor(x, spk, torch.zeros_like(x), 1.0, None)
        enc, proj = model.mel_encoder(x, spk, zero, zero, pitch, None)
        compare("encoder", encoded, enc + proj)
        for n in [61, 62, 63]:  # every remainder of the hourglass' factor of 3
            frames = torch.repeat_interleave(encoded[0], 3, 0)[None, :n]
            real, imag = decoder(frames)
            magnitude, cos, sin = model.vocoder(model.mel_decoder(frames), 48000, 0.0, True)
            compare(f"decoder ({n} frames)", torch.cat([real, imag]), torch.cat([magnitude * cos, magnitude * sin]))
        frames = torch.repeat_interleave(encoded[0], 4, 0)[None]

    os.makedirs(out, exist_ok=True)
    parts = [
        ("encoder", encoder, (x, spk), ["tokens", "speaker"], ["log_durations", "encoded"],
         {"tokens": {1: "symbols"}, "log_durations": {1: "symbols"}, "encoded": {1: "symbols"}}),
        ("decoder", decoder, (frames,), ["frames"], ["real", "imag"],
         {"frames": {1: "frames"}, "real": {2: "spectra"}, "imag": {2: "spectra"}}),
    ]
    for name, module, args, inputs, outputs, axes in parts:
        path = os.path.join(out, f"{name}.onnx")
        torch.onnx.export(module, args, path, opset_version=17, input_names=inputs, output_names=outputs,
                          dynamic_axes=axes, do_constant_folding=True)
        print(name, os.path.getsize(path), "bytes")
    meta = {"symbols": SYMBOLS, "bos": 2, "eos": 1, "sample_rate": 48000, "n_fft": N_FFT, "hop": HOP,
            "speakers": [f"{LANGUAGES[s.split('_')[0]]}:{s}" for s in SPEAKERS]}
    json.dump(meta, open(os.path.join(out, "silero.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def istft(real, imag):
    """Vocos ISTFT, padding "same": real/imag are (bins, frames)."""
    spec = real + 1j * imag
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
        self.encoder = ort.InferenceSession(os.path.join(directory, "encoder.onnx"), options)
        self.decoder = ort.InferenceSession(os.path.join(directory, "decoder.onnx"), options)

    def __call__(self, text, speaker):
        x = tokens(text)
        spk = np.array([speaker], dtype=np.int64)
        log, encoded = self.encoder.run(None, {"tokens": x, "speaker": spk})
        frames = np.round(np.exp(log[0]) - 1).astype(np.int64)
        frames[0] = min(frames[0], 5)
        frames[frames < 0] = 0
        expanded = np.repeat(encoded[0], frames, axis=0)[None].astype(np.float32)
        real, imag = self.decoder.run(None, {"frames": expanded})
        return istft(real[0], imag[0])


def check(jit, directory):
    model = load(jit)
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
