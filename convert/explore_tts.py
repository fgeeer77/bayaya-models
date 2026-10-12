#!/usr/bin/env python3
"""Looks inside the speech models that need an engine of their own (Silero cis_base, Silero Stress,
Vosk-TTS), so their port can be planned: structure, inputs and outputs, what ONNX export makes of
them. Prints only; uploads nothing.

    python3 explore_tts.py silero <v5_cis_base_nostress.jit>
    python3 explore_tts.py stress
    python3 explore_tts.py vosk <model dir>
"""
import glob
import json
import os
import sys
import time
import traceback


def section(title):
    print(f"\n========== {title} ==========", flush=True)


def silero(path):
    import torch

    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    section("forward code")
    print(model.code[:6000])
    section("children")
    for name, child in model.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"{name}: {child.original_name} params={params}")
        try:
            print("   ", child.code[:1500].replace("\n", "\n    "))
        except Exception as e:  # noqa: BLE001
            print("    (no code)", e)
    section("methods")
    print([m for m in dir(model) if not m.startswith("_")][:80])
    section("ops of interest")
    graph = str(model.inlined_graph)
    for op in ["istft", "stft", "fft", "irfft", "rfft", "fold", "unfold", "conv_transpose", "lstm", "gru", "multinomial", "rand", "normal"]:
        print(op, graph.count(op))
    section("run")
    symbols = "|||!'+,-.:;?hабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїјўґғҕҗҙқҝҡңҥҫүұҳҷҹһӑӗәӝӟӣӥӧөӯӱӳӵӏ—… "
    ids = {s: i for i, s in enumerate(symbols)}
    text = "в н+едрах т+ундры в+ыдры в г+етрах т+ырят в в+ёдра +ядра к+едров."
    x = torch.tensor([[2] + [ids[c] for c in text if c in ids] + [1]])
    for speaker in [26, 40]:
        started = time.time()
        with torch.no_grad():
            audio = model(x, torch.LongTensor([speaker]))
        took = time.time() - started
        print(f"speaker {speaker}: out {tuple(audio.shape)} {audio.dtype} seconds={audio.shape[-1] / 48000:.2f} took={took:.2f}s "
              f"min={audio.min():.3f} max={audio.max():.3f}")
    # Same input twice: is the output deterministic?
    with torch.no_grad():
        a = model(x, torch.LongTensor([26]))
        b = model(x, torch.LongTensor([26]))
    print("deterministic:", bool(torch.equal(a, b)), float((a - b).abs().max()) if a.shape == b.shape else "shapes differ")
    section("onnx export")
    for opset in [17, 18, 20]:
        try:
            torch.onnx.export(model, (x, torch.LongTensor([26])), f"/tmp/silero{opset}.onnx", opset_version=opset,
                              input_names=["tokens", "speaker"], output_names=["audio"],
                              dynamic_axes={"tokens": {1: "T"}, "audio": {0: "N"}})
            print(f"opset {opset}: OK, {os.path.getsize(f'/tmp/silero{opset}.onnx')} bytes")
            break
        except Exception as e:  # noqa: BLE001
            print(f"opset {opset}: {type(e).__name__}: {str(e)[:1500]}")
    section("lite interpreter")
    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        lite = optimize_for_mobile(model)
        lite._save_for_lite_interpreter("/tmp/silero.ptl")
        print("ptl bytes:", os.path.getsize("/tmp/silero.ptl"))
    except Exception as e:  # noqa: BLE001
        print(f"{type(e).__name__}: {str(e)[:1500]}")
        try:
            model._save_for_lite_interpreter("/tmp/silero-plain.ptl")
            print("plain ptl bytes:", os.path.getsize("/tmp/silero-plain.ptl"))
        except Exception as e2:  # noqa: BLE001
            print(f"plain: {type(e2).__name__}: {str(e2)[:800]}")


def stress():
    import silero_stress

    section("silero_stress package")
    root = os.path.dirname(silero_stress.__file__)
    for p in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            print(os.path.relpath(p, root), os.path.getsize(p))
    print([n for n in dir(silero_stress) if not n.startswith("_")])
    for name in sorted(glob.glob(os.path.join(root, "*.py"))):
        section(os.path.basename(name))
        print(open(name, encoding="utf-8").read()[:5000])
    section("use")
    try:
        accentor = silero_stress.load_accentor()
        print(type(accentor))
        for text in ["Мой друг купил новый замок для двери и поехал смотреть старинный замок.",
                     "Ёжик сидел на берегу озера и смотрел на облака."]:
            print(accentor(text))
        print([m for m in dir(accentor) if not m.startswith("__")][:60])
    except Exception:  # noqa: BLE001
        traceback.print_exc()


def vosk(directory):
    import onnx

    section("files")
    for p in sorted(glob.glob(os.path.join(directory, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            print(os.path.relpath(p, directory), os.path.getsize(p))
    config = json.load(open(os.path.join(directory, "config.json"), encoding="utf-8"))
    section("config")
    print(json.dumps({k: v for k, v in config.items() if k != "phoneme_id_map"}, ensure_ascii=False)[:3000])
    print("phonemes:", len(config.get("phoneme_id_map", {})))
    for name in ["model.onnx", "bert/model.onnx"]:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        model = onnx.load(path, load_external_data=False)
        section(name)
        for i in model.graph.input:
            print("in ", i.name, [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim], i.type.tensor_type.elem_type)
        for o in model.graph.output:
            print("out", o.name, [d.dim_param or d.dim_value for d in o.type.tensor_type.shape.dim])
        print("meta", {p.key: p.value[:200] for p in model.metadata_props})
    section("dictionary sample")
    with open(os.path.join(directory, "dictionary"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 10 or i % 400000 == 0:
                print(repr(line))
        print("lines:", i + 1)


def silero_parts(path):
    """Second look: the glue code, the ISTFT settings, and an ONNX export of each network on its own."""
    import numpy as np
    import onnxruntime as ort
    import torch

    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    section("constants")
    code, constants = model.code_with_constants
    print({k: (v if not hasattr(v, "shape") or v.numel() < 20 else tuple(v.shape)) for k, v in constants.const_mapping.items()})
    for name in ["preprocess_durs", "fx_pauses"]:
        section(name)
        try:
            print(getattr(model, name).code[:4000])
        except Exception as e:  # noqa: BLE001
            print(e)
    section("istft")
    istft = model.istft
    for attr in ["n_fft", "hop_length", "win_length", "padding"]:
        print(attr, getattr(istft, attr))
    w = istft.window
    print("window", tuple(w.shape), w[:5].tolist(), float(w.sum()), float((w ** 2).sum()))
    print(istft.code[1500:4000])
    section("vocoder head")
    print(model.vocoder.head.code[:3000])
    section("dur_pred")
    print(model.dur_predictor.dur_pred.code[:2500])
    section("mel_encoder rest")
    print(model.mel_encoder.code[1500:4000])

    symbols = "|||!'+,-.:;?hабвгдежзийклмнопрстуфхцчшщъыьэюяёєіїјўґғҕҗҙқҝҡңҥҫүұҳҷҹһӑӗәӝӟӣӥӧөӯӱӳӵӏ—… "
    ids = {s: i for i, s in enumerate(symbols)}
    text = "в н+едрах т+ундры в+ыдры в г+етрах т+ырят в в+ёдра +ядра к+едров."
    x = torch.tensor([[2] + [ids[c] for c in text if c in ids] + [1]])
    spk = torch.LongTensor([26])
    with torch.no_grad():
        reference = model(x, spk)
        seq, symb = model.preprocess_durs(x)
        print("preprocess:", tuple(x.shape), "->", tuple(seq.shape), "symb_durs", symb)
        log_dur = model.dur_predictor(seq, spk, None, 1.0, None)
        dur = torch.round(torch.exp(log_dur) - 1).squeeze(0).long()
        print("log_dur", tuple(log_dur.shape), "dur", dur.tolist())
        pitch = model.pitch_predictor(seq, spk, dur.unsqueeze(0), 1.0, None)
        lang = constants.const_mapping.get("c0")
        enc, proj = model.mel_encoder(seq, spk, lang, torch.tensor([0]), pitch, None)
        print("pitch", tuple(pitch.shape), "enc", tuple(enc.shape), "proj", tuple(proj.shape))
        dur0 = dur.clone()
        if dur0[0] > 5:
            dur0[0] = 5
        dur0[dur0 < 0] = 0
        expanded = torch.repeat_interleave(enc[0], dur0, 0)[None] + torch.repeat_interleave(proj[0], dur0, 0)[None]
        mel = model.mel_decoder(expanded)
        mag, re, im = model.vocoder(mel, 48000, 0.0, True)
        print("mel", tuple(mel.shape), "mag", tuple(mag.shape))
        audio = model.istft(mag * torch.complex(re, im))
        print("by parts vs whole:", tuple(audio.shape), tuple(reference.shape),
              float((audio[..., :reference.shape[-1]] - reference).abs().max()) if audio.shape == reference.shape else "shapes differ")

    class Durations(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, tokens, speaker):
            return self.m.dur_predictor(tokens, speaker, None, 1.0, None)

    class Encoder(torch.nn.Module):
        def __init__(self, m, lang):
            super().__init__()
            self.m = m
            self.lang = lang

        def forward(self, tokens, speaker, durations):
            pitch = self.m.pitch_predictor(tokens, speaker, durations, 1.0, None)
            return self.m.mel_encoder(tokens, speaker, self.lang, torch.tensor([0]), pitch, None)

    class Decoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, expanded):
            mel = self.m.mel_decoder(expanded)
            return self.m.vocoder(mel, 48000, 0.0, True)

    parts = {
        "durations": (Durations(model), (seq, spk), ["tokens", "speaker"], ["log_durations"], {"tokens": {1: "T"}, "log_durations": {1: "T"}}),
        "encoder": (Encoder(model, lang), (seq, spk, dur0.unsqueeze(0)), ["tokens", "speaker", "durations"], ["encoded", "pitch"],
                    {"tokens": {1: "T"}, "durations": {1: "T"}, "encoded": {1: "T"}, "pitch": {1: "T"}}),
        "decoder": (Decoder(model), (expanded,), ["frames"], ["magnitude", "real", "imag"],
                    {"frames": {1: "F"}, "magnitude": {2: "F2"}, "real": {2: "F2"}, "imag": {2: "F2"}}),
    }
    section("export by parts")
    for name, (module, args, inputs, outputs, axes) in parts.items():
        for opset in [17]:
            try:
                torch.onnx.export(module, args, f"/tmp/{name}.onnx", opset_version=opset, input_names=inputs,
                                  output_names=outputs, dynamic_axes=axes)
                session = ort.InferenceSession(f"/tmp/{name}.onnx")
                got = session.run(None, {i: a.numpy() for i, a in zip(inputs, args)})
                with torch.no_grad():
                    want = module(*args)
                want = want if isinstance(want, (tuple, list)) else [want]
                diffs = [float(np.abs(g - w.numpy()).max()) for g, w in zip(got, want)]
                print(f"{name}: OK {os.path.getsize(f'/tmp/{name}.onnx')} bytes, max diff {diffs}")
            except Exception as e:  # noqa: BLE001
                print(f"{name}: {type(e).__name__}: {str(e)[:2000]}")
    # Other lengths: does the exported graph follow the input length?
    try:
        x2 = torch.tensor([[2] + [ids[c] for c in "пр+ивет." if c in ids] + [1]])
        seq2, _ = model.preprocess_durs(x2)
        s = ort.InferenceSession("/tmp/durations.onnx")
        got = s.run(None, {"tokens": seq2.numpy(), "speaker": spk.numpy()})[0]
        with torch.no_grad():
            want = model.dur_predictor(seq2, spk, None, 1.0, None).numpy()
        print("durations on another length:", got.shape, float(np.abs(got - want).max()))
    except Exception as e:  # noqa: BLE001
        print("other length:", e)


def stress_parts():
    """How Silero Stress is built: the package's own sources and its pieces."""
    import torch
    import silero_stress

    root = os.path.dirname(silero_stress.__file__)
    importer = torch.package.PackageImporter(os.path.join(root, "data", "accentor.pt"))
    section("package files")
    print(importer.file_structure())
    for name in sorted(n for n in importer.zip_reader.get_all_records() if n.endswith(".py")):
        section(name)
        try:
            print(importer.zip_reader.get_record(name).decode("utf-8")[:6000])
        except Exception as e:  # noqa: BLE001
            print(e)
    accentor = silero_stress.load_accentor()
    section("pieces")
    for name in ["accentor", "homosolver"]:
        part = getattr(accentor, name)
        print(name, type(part), [a for a in dir(part) if not a.startswith("__")][:50])
        for attr in dir(part):
            if attr.startswith("__"):
                continue
            value = getattr(part, attr, None)
            if isinstance(value, (dict, list, set)):
                print(f"  {attr}: {type(value).__name__} of {len(value)}", list(value)[:5] if not isinstance(value, dict) else list(value.items())[:5])
            elif isinstance(value, torch.nn.Module):
                print(f"  {attr}: module params={sum(p.numel() for p in value.parameters())}")


def silero_tree(path):
    """Every module of the networks: class, parameter shapes, and the code of each non-standard class once."""
    import torch

    model = torch.jit.load(path, map_location="cpu")
    model.eval()
    section("tree")
    seen = {}
    for name, module in model.named_modules():
        cls = module.original_name
        own = {n: tuple(p.shape) for n, p in module.named_parameters(recurse=False)}
        buffers = {n: tuple(b.shape) for n, b in module.named_buffers(recurse=False)}
        consts = {}
        for attr in ["num_heads", "embed_dim", "batch_first", "kernel_size", "padding", "dilation", "stride", "groups",
                     "eps", "normalized_shape", "pitch_strength", "sil_value", "point_id", "articulation_emb", "_qkv_same_embed_dim",
                     "head_dim", "dropout", "in_channels", "out_channels", "padding_mode", "approximate", "dim", "scale",
                     "n_layers", "d_model", "pre_lnorm"]:
            try:
                v = getattr(module, attr)
                if not isinstance(v, torch.nn.Module) and not callable(v):
                    consts[attr] = v if not isinstance(v, torch.Tensor) or v.numel() < 4 else tuple(v.shape)
            except Exception:  # noqa: BLE001
                pass
        print(f"{name or '<root>'}: {cls} {own} {buffers} {consts}")
        if cls not in seen:
            seen[cls] = module
    standard = {"Linear", "Conv1d", "Embedding", "LayerNorm", "Dropout", "ReLU", "GELU", "Sequential", "ModuleList",
                "MultiheadAttention", "NonDynamicallyQuantizableLinear", "Identity", "Tanh", "LeakyReLU", "SiLU", "Mish",
                "BatchNorm1d", "ConvTranspose1d", "Softplus"}
    for cls, module in seen.items():
        if cls in standard:
            continue
        section(f"class {cls}")
        for method in ["forward"] + [m for m in dir(module) if not m.startswith("_") and m not in ("forward",)]:
            try:
                fn = getattr(module, method)
                if hasattr(fn, "code") and method != "forward":
                    print(f"--- {method}")
                    print(fn.code[:3000])
            except Exception:  # noqa: BLE001
                pass
        try:
            print(module.code[:5000])
        except Exception as e:  # noqa: BLE001
            print("(no code)", e)
    section("lite interpreter of scripted parts")
    try:
        class Durations(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.durations = m.dur_predictor

            def forward(self, tokens: torch.Tensor, speaker: torch.Tensor) -> torch.Tensor:
                return self.durations(tokens, speaker, None, 1.0, None)

        scripted = torch.jit.script(Durations(model))
        scripted._save_for_lite_interpreter("/tmp/durations.ptl")
        print("durations.ptl", os.path.getsize("/tmp/durations.ptl"))
    except Exception as e:  # noqa: BLE001
        print(f"{type(e).__name__}: {str(e)[:1000]}")


if __name__ == "__main__":
    {"silero": lambda: silero(sys.argv[2]), "stress": stress, "vosk": lambda: vosk(sys.argv[2]),
     "silero-parts": lambda: silero_parts(sys.argv[2]), "stress-parts": stress_parts,
     "silero-tree": lambda: silero_tree(sys.argv[2])}[sys.argv[1]]()
