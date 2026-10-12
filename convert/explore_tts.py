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


if __name__ == "__main__":
    {"silero": lambda: silero(sys.argv[2]), "stress": stress, "vosk": lambda: vosk(sys.argv[2])}[sys.argv[1]]()
