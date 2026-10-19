#!/usr/bin/env python3
"""Checks mt_seq2seq.py's decoder step and export on tiny randomly initialised M2M100 and T5 models
(nothing is downloaded): greedy decoding with our step equals transformers' generate(), the exported
fp32 graphs decode the same in ONNX Runtime (with an empty cache on the first step), and the 8-bit and
4-bit quantizers produce graphs that run.

    python3 test_mt_seq2seq.py
"""
import os
import random
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_seq2seq as mt  # noqa: E402

MAX_NEW = 24


def tiny(family):
    from transformers import M2M100Config, M2M100ForConditionalGeneration, T5Config, T5ForConditionalGeneration
    torch.manual_seed(0)
    if family == "m2m":
        cfg = M2M100Config(vocab_size=500, d_model=64, encoder_layers=2, decoder_layers=3, encoder_attention_heads=4,
                           decoder_attention_heads=4, encoder_ffn_dim=128, decoder_ffn_dim=128, max_position_embeddings=128,
                           scale_embedding=True, pad_token_id=1, bos_token_id=0, eos_token_id=2, decoder_start_token_id=2,
                           tie_word_embeddings=True, dropout=0.0, attention_dropout=0.0, activation_dropout=0.0)
        model = M2M100ForConditionalGeneration(cfg)
    else:
        cfg = T5Config(vocab_size=500, d_model=64, d_kv=16, d_ff=128, num_layers=2, num_decoder_layers=3, num_heads=4,
                       feed_forward_proj="gated-gelu", tie_word_embeddings=False, pad_token_id=0, eos_token_id=1,
                       decoder_start_token_id=0, dropout_rate=0.0)
        model = T5ForConditionalGeneration(cfg)
    model.config._attn_implementation = "eager"
    # Random weights decode the same token forever; larger output weights make sequences varied.
    with torch.no_grad():
        model.lm_head.weight.mul_(8.0)
    return model.eval()


def generate(model, family, enc_ids, forced):
    from transformers import GenerationConfig
    cfg = GenerationConfig(num_beams=1, do_sample=False, max_new_tokens=MAX_NEW, decoder_start_token_id=model.config.decoder_start_token_id,
                           eos_token_id=model.config.eos_token_id, pad_token_id=model.config.pad_token_id,
                           forced_bos_token_id=forced if family == "m2m" else None)
    with torch.no_grad():
        return model.generate(torch.tensor([enc_ids]), generation_config=cfg)[0].tolist()


def run(family):
    model = tiny(family)
    enc, dec = mt.wrappers(model, family)
    d = mt.dims(model, family)
    eos = model.config.eos_token_id
    start = model.config.decoder_start_token_id
    rnd = random.Random(1)
    cases = []
    for n in [1, 3, 7, 15, 30]:
        enc_ids = [rnd.randrange(5, 500) for _ in range(n)] + [eos]
        forced = rnd.randrange(5, 500)
        prefix = [start, forced] if family == "m2m" else [start]
        cases.append((enc_ids, forced, prefix))
    for enc_ids, forced, prefix in cases:
        ref = generate(model, family, enc_ids, forced)
        ours = mt.greedy_torch(enc, dec, enc_ids, prefix, eos, d["layers"], d["heads"], d["headDim"], MAX_NEW)
        # transformers counts the forced token in max_new_tokens; compare the common length.
        n = min(len(ref), len(ours))
        assert ours[:n] == ref[:n], f"{family}: our step differs from generate()\n ours {ours}\n hf   {ref}"
    print(f"{family}: our step = generate() on {len(cases)} inputs")

    with tempfile.TemporaryDirectory() as tmp:
        mt.export_graphs(enc, dec, d, cases[2][0], cases[2][2], tmp)
        os.makedirs(f"{tmp}/fp32", exist_ok=True)
        for part in ["encoder", "decoder"]:
            os.replace(f"{tmp}/{part}/{part}.onnx", f"{tmp}/fp32/{part}.onnx")
        graphs = mt.OrtGraphs(f"{tmp}/fp32", d["layers"], d["heads"], d["headDim"], eos, MAX_NEW, threads=1)
        for enc_ids, forced, prefix in cases:
            ours = mt.greedy_torch(enc, dec, enc_ids, prefix, eos, d["layers"], d["heads"], d["headDim"], MAX_NEW)
            onnx_ids = graphs.greedy(enc_ids, prefix)
            assert onnx_ids == ours, f"{family}: onnx differs\n onnx  {onnx_ids}\n torch {ours}"
        print(f"{family}: fp32 onnx = torch on {len(cases)} inputs")
        for bits in ([8] if family == "m2m" else [8, 4]):
            os.makedirs(f"{tmp}/q{bits}", exist_ok=True)
            for part in ["encoder", "decoder"]:
                mt.quantize(f"{tmp}/fp32/{part}.onnx", f"{tmp}/q{bits}/{part}.onnx", bits)
            q = mt.OrtGraphs(f"{tmp}/q{bits}", d["layers"], d["heads"], d["headDim"], eos, MAX_NEW, threads=1)
            same = sum(q.greedy(e, p) == graphs.greedy(e, p) for e, _, p in cases)
            sizes = {part: os.path.getsize(f"{tmp}/q{bits}/{part}.onnx") for part in ["encoder", "decoder"]}
            ops = sorted({n.op_type for part in ["encoder", "decoder"] for n in __import__("onnx").load(f"{tmp}/q{bits}/{part}.onnx").graph.node
                          if n.op_type in ("MatMulInteger", "DynamicQuantizeMatMul", "MatMulNBits", "GatherBlockQuantized", "DequantizeLinear")})
            print(f"{family}: {bits}-bit runs, same as fp32 on {same}/{len(cases)}, sizes {sizes}, ops {ops}")


if __name__ == "__main__":
    for family in ["m2m", "t5"]:
        run(family)
