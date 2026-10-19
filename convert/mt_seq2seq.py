#!/usr/bin/env python3
"""Big multilingual translation models for Bayaya Platinum, exported for ONNX Runtime in the one shape
the app runs them all with:

- NLLB-200 distilled 600M and 1.3B, M2M100 418M (the M2M100 architecture, Meta);
- MADLAD-400 3B MT (T5, Google).

encoder.onnx   input_ids [1, S], attention_mask [1, S]
               -> cross_k, cross_v [L, 1, H, S, Dk]: every decoder layer's cross-attention keys and
                  values, computed once per sentence.
decoder.onnx   input_ids [1, T], encoder_attention_mask [1, S], cross_k, cross_v,
               self_k, self_v [L, 1, H, P, Dk] (P = 0 on the first step)
               -> logits [1, V] of the last position, new_self_k, new_self_v [L, 1, H, P + T, Dk].

The decoder graph is our own step written from the model's modules (as transformers 4.46 runs them);
`export` checks that greedy decoding with it gives exactly what transformers' generate() gives. The
graphs are quantized: 8-bit weights (NLLB, M2M100) or 4-bit blocks (MADLAD). Next to them:
sentencepiece.model, ids.bin (model id of every SentencePiece id, int32 little-endian) and model.json
(sizes, special ids, and per language the ids around the text), so the app tokenizes without Python.

    python3 mt_seq2seq.py export nllb-600m out/       # downloads the model from Hugging Face
    python3 mt_seq2seq.py check  nllb-600m out/ flores200_dataset
"""
import gc
import json
import math
import os
import resource
import shutil
import struct
import sys
import time

import langcodes
import numpy as np
import torch
from torch import nn

MODELS = {
    "nllb-600m": {"repo": "facebook/nllb-200-distilled-600M", "family": "m2m", "bits": 8, "title": "NLLB-200 600M",
                  "license": "CC-BY-NC-4.0"},
    "nllb-1.3b": {"repo": "facebook/nllb-200-distilled-1.3B", "family": "m2m", "bits": 8, "title": "NLLB-200 1.3B",
                  "license": "CC-BY-NC-4.0"},
    "m2m100-418m": {"repo": "facebook/m2m100_418M", "family": "m2m", "bits": 8, "title": "M2M100 418M", "license": "MIT"},
    "madlad-3b": {"repo": "google/madlad400-3b-mt", "family": "t5", "bits": 4, "title": "MADLAD-400 3B", "license": "Apache-2.0"},
}

# Sentences the export checks decoding on (greedy, our step against transformers'), and the
# tokenizer test vectors the app's tests compare its own tokenization with.
SAMPLES = [
    ("en", "ru", "The weather is nice today, let's go for a walk in the park."),
    ("ru", "en", "Завтра в девять утра у меня встреча с врачом, не забудь напомнить."),
    ("ru", "uk", "Я не понимаю, почему поезд опять опаздывает на полчаса."),
    ("en", "ka", "Where is the nearest pharmacy?"),
    ("ru", "de", "Пожалуйста, пришлите мне счёт до конца недели."),
    ("de", "fr", "Die Katze schläft auf dem warmen Fensterbrett."),
    ("en", "zh", "Machine translation on a phone works without the internet."),
    ("ru", "en", "Цена: 1 299,50 ₽ (скидка 15 %) — до 31.12."),
]
TOKENIZER_TEXTS = [
    "Hello, world!", "Привет, мир! Как дела?", "  двойной  пробел и\tтаб", "გამარჯობა", "机器翻译", "Ünïcödé — «кавычки» 😀",
    "1 299,50 ₽", "e-mail: test@example.com", "",
]

# FLORES-200 devtest directions for the quality check (FLORES codes); a model skips what it lacks.
DIRECTIONS = [("eng_Latn", "rus_Cyrl"), ("rus_Cyrl", "eng_Latn"), ("rus_Cyrl", "ukr_Cyrl"), ("eng_Latn", "kat_Geor"),
              ("rus_Cyrl", "deu_Latn"), ("eng_Latn", "swh_Latn"), ("kaz_Cyrl", "rus_Cyrl"), ("tat_Cyrl", "rus_Cyrl")]
FLORES_SENTENCES = 50
MAX_NEW = 256
NEG = -1e9


# ---------------------------------------------------------------------------------------------
# Language codes: the model's own -> the app's (ISO 639-1 where there is one, else 639-3; a script
# other than the language's usual one after an underscore, lowercase: "ks_deva", "zh_hant").

def app_code(code, macro=True, lone=False):
    """[lone]: the model has this language in one script only (then an unknown usual script is dropped)."""
    tag = langcodes.standardize_tag(code.replace("_", "-"), macro=macro)
    lang = langcodes.Language.get(tag)
    if lang.territory:  # Dari ("fa-AF") and the like: the model's own language code
        lang = langcodes.Language.make(language=code.split("_")[0].lower(), script=lang.script)
    base, script = lang.language, lang.script
    if script:
        guess = langcodes.Language.get(base).maximize()
        # Languages CLDR does not know come back as "Latn, US" (its default): no usual script known.
        likely = None if (guess.script, guess.territory) == ("Latn", "US") and base != "en" else guess.script
        if script == likely or (base == "ko" and script == "Hang") or (likely is None and lone):
            script = None
    base = {"fil": "tl"}.get(base, base)
    return base + ("_" + script.lower() if script else "")


def language_table(codes):
    """Model codes -> app codes; when two map to one, the second keeps its own (non-macro) code."""
    bases = {}
    for c in codes:
        bases[c.split("_")[0]] = bases.get(c.split("_")[0], 0) + 1
    out, seen = {}, set()
    for c in codes:
        lone = bases[c.split("_")[0]] == 1
        a = app_code(c, lone=lone)
        if a in seen:
            a = app_code(c, macro=False, lone=lone)
        if a in seen:
            a = c.lower()
        seen.add(a)
        out[c] = a
    return out


FLORES_TO_APP = {f: app_code(f) for d in DIRECTIONS for f in d}


# ---------------------------------------------------------------------------------------------
# Loading

def load(model_id):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    spec = MODELS[model_id]
    model = AutoModelForSeq2SeqLM.from_pretrained(spec["repo"], torch_dtype=torch.float32, low_cpu_mem_usage=True,
                                                  attn_implementation="eager")
    model.eval()
    tok = AutoTokenizer.from_pretrained(spec["repo"], use_fast=False)
    return model, tok


def revision(model_id):
    from huggingface_hub import HfApi
    return HfApi().model_info(MODELS[model_id]["repo"]).sha


def split(x, heads, dk):
    return x.view(1, -1, heads, dk).transpose(1, 2)


def merge(x):
    return x.transpose(1, 2).reshape(1, x.shape[2], -1)


def attend(q, k, v, bias):
    scores = torch.matmul(q, k.transpose(-1, -2)) + bias
    return torch.matmul(torch.softmax(scores.float(), dim=-1), v)


def causal(q_len, past):
    """[1, 1, T, P + T] additive mask: query i (at position past + i) sees keys up to itself."""
    k_pos = torch.arange(past + q_len)[None, :]
    q_pos = (torch.arange(q_len) + past)[:, None]
    return ((k_pos > q_pos).float() * NEG)[None, None]


def enc_bias(mask):
    return ((1.0 - mask[:, None, None, :].float()) * NEG)


# ---------------------------------------------------------------------------------------------
# M2M100 / NLLB

class M2MEncoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.model.encoder
        self.layers = model.model.decoder.layers
        self.heads = model.config.decoder_attention_heads
        self.dk = model.config.d_model // self.heads

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        ks = [split(layer.encoder_attn.k_proj(h), self.heads, self.dk) for layer in self.layers]
        vs = [split(layer.encoder_attn.v_proj(h), self.heads, self.dk) for layer in self.layers]
        return torch.stack(ks), torch.stack(vs)


class M2MDecoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        dec = model.model.decoder
        assert model.config.tie_word_embeddings
        # One matrix for both ends: the embedding is its columns (Gather on axis 1), the output layer
        # multiplies by it, so the quantized graph keeps a single copy of the vocabulary.
        self.wt = nn.Parameter(model.lm_head.weight.detach().t().contiguous(), requires_grad=False)
        self.scale = dec.embed_tokens.embed_scale
        self.positions = dec.embed_positions.weights
        self.padding = dec.embed_positions.padding_idx
        self.layers = dec.layers
        self.norm = dec.layer_norm
        self.heads = model.config.decoder_attention_heads
        self.dk = model.config.d_model // self.heads

    def forward(self, input_ids, encoder_attention_mask, cross_k, cross_v, self_k, self_v):
        past = self_k.shape[3]
        q_len = input_ids.shape[1]
        emb = torch.index_select(self.wt, 1, input_ids.view(-1)).t().unsqueeze(0) * self.scale
        pos = torch.arange(q_len) + past + self.padding + 1
        h = emb + self.positions.index_select(0, pos).unsqueeze(0)
        self_bias = causal(q_len, past)
        cross_bias = enc_bias(encoder_attention_mask)
        new_k, new_v = [], []
        for i, layer in enumerate(self.layers):
            a = layer.self_attn
            x = layer.self_attn_layer_norm(h)
            q = split(a.q_proj(x) * a.scaling, self.heads, self.dk)
            k = torch.cat([self_k[i], split(a.k_proj(x), self.heads, self.dk)], dim=2)
            v = torch.cat([self_v[i], split(a.v_proj(x), self.heads, self.dk)], dim=2)
            new_k.append(k)
            new_v.append(v)
            h = h + a.out_proj(merge(attend(q, k, v, self_bias)))
            c = layer.encoder_attn
            x = layer.encoder_attn_layer_norm(h)
            q = split(c.q_proj(x) * c.scaling, self.heads, self.dk)
            h = h + c.out_proj(merge(attend(q, cross_k[i], cross_v[i], cross_bias)))
            x = layer.final_layer_norm(h)
            h = h + layer.fc2(layer.activation_fn(layer.fc1(x)))
        h = self.norm(h[:, -1, :])
        return torch.matmul(h, self.wt), torch.stack(new_k), torch.stack(new_v)


# ---------------------------------------------------------------------------------------------
# T5 / MADLAD

class T5Encoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder
        self.blocks = model.decoder.block
        self.heads = model.config.num_heads
        self.dk = model.config.d_kv

    def forward(self, input_ids, attention_mask):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        ks = [split(b.layer[1].EncDecAttention.k(h), self.heads, self.dk) for b in self.blocks]
        vs = [split(b.layer[1].EncDecAttention.v(h), self.heads, self.dk) for b in self.blocks]
        return torch.stack(ks), torch.stack(vs)


class T5Decoder(nn.Module):
    def __init__(self, model):
        super().__init__()
        cfg = model.config
        self.embed = model.decoder.embed_tokens
        self.blocks = model.decoder.block
        self.norm = model.decoder.final_layer_norm
        self.lm_head = model.lm_head
        self.rel = self.blocks[0].layer[0].SelfAttention
        self.out_scale = cfg.d_model ** -0.5 if cfg.tie_word_embeddings else 1.0
        self.heads = cfg.num_heads
        self.dk = cfg.d_kv

    def forward(self, input_ids, encoder_attention_mask, cross_k, cross_v, self_k, self_v):
        from transformers.models.t5.modeling_t5 import T5Attention
        past = self_k.shape[3]
        q_len = input_ids.shape[1]
        h = self.embed(input_ids)
        rel = torch.arange(past + q_len)[None, :] - (torch.arange(q_len) + past)[:, None]
        buckets = T5Attention._relative_position_bucket(rel, bidirectional=False, num_buckets=self.rel.relative_attention_num_buckets,
                                                         max_distance=self.rel.relative_attention_max_distance)
        self_bias = self.rel.relative_attention_bias(buckets).permute(2, 0, 1).unsqueeze(0) + causal(q_len, past)
        cross_bias = enc_bias(encoder_attention_mask)
        new_k, new_v = [], []
        for i, block in enumerate(self.blocks):
            sa = block.layer[0]
            x = sa.layer_norm(h)
            a = sa.SelfAttention
            q = split(a.q(x), self.heads, self.dk)
            k = torch.cat([self_k[i], split(a.k(x), self.heads, self.dk)], dim=2)
            v = torch.cat([self_v[i], split(a.v(x), self.heads, self.dk)], dim=2)
            new_k.append(k)
            new_v.append(v)
            h = h + a.o(merge(attend(q, k, v, self_bias)))
            ca = block.layer[1]
            x = ca.layer_norm(h)
            q = split(ca.EncDecAttention.q(x), self.heads, self.dk)
            h = h + ca.EncDecAttention.o(merge(attend(q, cross_k[i], cross_v[i], cross_bias)))
            h = block.layer[2](h)
        h = self.norm(h[:, -1:, :]) * self.out_scale
        return self.lm_head(h)[:, 0, :], torch.stack(new_k), torch.stack(new_v)


def wrappers(model, family):
    return (M2MEncoder(model), M2MDecoder(model)) if family == "m2m" else (T5Encoder(model), T5Decoder(model))


def dims(model, family):
    c = model.config
    if family == "m2m":
        return {"layers": c.decoder_layers, "heads": c.decoder_attention_heads, "headDim": c.d_model // c.decoder_attention_heads,
                "vocab": c.vocab_size}
    return {"layers": c.num_decoder_layers, "heads": c.num_heads, "headDim": c.d_kv, "vocab": c.vocab_size}


# ---------------------------------------------------------------------------------------------
# Tokenization as the app does it: SentencePiece ids -> model ids (ids.bin), the language's ids
# around them (model.json).

class Spec:
    """What model.json says, read back: the app's view of tokenization and decoding."""

    def __init__(self, folder):
        import sentencepiece as spm
        self.meta = json.load(open(os.path.join(folder, "model.json")))
        self.sp = spm.SentencePieceProcessor(model_file=os.path.join(folder, "sentencepiece.model"))
        raw = open(os.path.join(folder, "ids.bin"), "rb").read()
        self.ids = list(struct.unpack(f"<{len(raw) // 4}i", raw))
        self.back = {}
        for spm_id, model_id in enumerate(self.ids):
            self.back.setdefault(model_id, spm_id)
        self.langs = {lang["code"]: lang for lang in self.meta["languages"]}

    def encode(self, text, src, tgt):
        m = self.meta
        body = [self.ids[i] for i in self.sp.encode(text)][: m["maxInput"]]
        return self.langs[src]["sourcePrefix"] + self.langs[tgt]["targetEncoderPrefix"] + body + [m["eos"]]

    def decoder_prefix(self, tgt):
        return [self.meta["decoderStart"]] + self.langs[tgt]["targetDecoderPrefix"]

    def decode(self, ids):
        skip = set(self.meta["special"])
        return self.sp.decode([self.back[i] for i in ids if i not in skip and i in self.back])


def hf_encode(tok, family, text, src_code, tgt_code):
    """What transformers' own tokenizer gives for the encoder (the reference)."""
    if family == "m2m":
        tok.src_lang = src_code
        return tok(text).input_ids
    return tok(f"{tgt_code} {text}").input_ids


def tokenizer_files(model_id, model, tok, folder):
    spec = MODELS[model_id]
    family = spec["family"]
    sp_path = getattr(tok, "spm_file", None) or tok.vocab_file
    shutil.copy(sp_path, os.path.join(folder, "sentencepiece.model"))
    sp = tok.sp_model
    ids = []
    for i in range(sp.get_piece_size()):
        ids.append(int(tok.convert_tokens_to_ids(sp.id_to_piece(i))))
    with open(os.path.join(folder, "ids.bin"), "wb") as f:
        f.write(struct.pack(f"<{len(ids)}i", *ids))

    if family == "m2m":
        if "nllb" in model_id:
            from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
            codes = list(FAIRSEQ_LANGUAGE_CODES)
            token = lambda c: c  # noqa: E731
        else:
            from transformers.models.m2m_100.tokenization_m2m_100 import FAIRSEQ_LANGUAGE_CODES
            codes = list(FAIRSEQ_LANGUAGE_CODES["m2m100"])
            token = lambda c: f"__{c}__"  # noqa: E731
        fixes = {"ns": "nso"}
        table = {c: fixes.get(c) or v for c, v in language_table(codes).items()}
        languages = []
        for c in codes:
            lid = tok.convert_tokens_to_ids(token(c))
            languages.append({"code": table[c], "token": token(c), "sourcePrefix": [lid], "targetEncoderPrefix": [],
                              "targetDecoderPrefix": [lid]})
        special = sorted({tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id} |
                         {lang["sourcePrefix"][0] for lang in languages})
    else:
        pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
        tags = [p[2:-1] for p in pieces if p.startswith("<2") and p.endswith(">")]
        valid = [t for t in tags if langcodes.tag_is_valid(t.replace("_", "-")) and not t.startswith(("zxx", "und"))]
        table = language_table(valid)
        languages = []
        for t in valid:
            prefix = tok(f"<2{t}>", add_special_tokens=False).input_ids
            languages.append({"code": table[t], "token": f"<2{t}>", "sourcePrefix": [], "targetEncoderPrefix": prefix,
                              "targetDecoderPrefix": []})
        special = sorted({tok.pad_token_id, tok.eos_token_id, tok.unk_token_id})
    cfg = model.config
    meta = {
        "id": model_id, "title": spec["title"], "family": family, "license": spec["license"], "source": f"https://huggingface.co/{spec['repo']}",
        **dims(model, family),
        "eos": int(tok.eos_token_id), "pad": int(tok.pad_token_id), "decoderStart": int(cfg.decoder_start_token_id),
        "unk": int(tok.unk_token_id), "special": [int(s) for s in special],
        # Longest encoder input the app feeds (longer text is split into sentences first).
        "maxInput": 400, "maxNew": MAX_NEW,
        "languages": languages,
    }
    json.dump(meta, open(os.path.join(folder, "model.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{len(languages)} languages:", " ".join(f"{lang['token']}={lang['code']}" for lang in languages))

    # Test vectors for the app's tokenizer, and the check that model.json reproduces transformers.
    s = Spec(folder)
    by_code = {lang["code"]: lang for lang in languages}
    vectors, mismatches = [], 0
    for text in TOKENIZER_TEXTS + [t for _, _, t in SAMPLES]:
        spm_ids = sp.encode(text)
        vectors.append({"text": text, "spm": spm_ids, "model": [ids[i] for i in spm_ids],
                        "decoded": s.decode([ids[i] for i in spm_ids])})
    for src, tgt, text in SAMPLES:
        if src not in by_code or tgt not in by_code:
            continue
        ours = s.encode(text, src, tgt)
        ref = hf_encode(tok, family, text, by_code[src]["token"], by_code[tgt]["token"])
        if ours != ref:
            mismatches += 1
            print(f"TOKENIZER MISMATCH {src}->{tgt} {text!r}\n  ours {ours}\n  hf   {ref}")
        vectors.append({"text": text, "source": src, "target": tgt, "encoder": ours})
    json.dump({"vectors": vectors}, open(os.path.join(folder, "tokenizer-tests.json"), "w"), ensure_ascii=False, indent=1)
    return mismatches


# ---------------------------------------------------------------------------------------------
# Decoding

def greedy_torch(enc, dec, enc_ids, prefix, eos, layers, heads, dk, max_new=MAX_NEW):
    ids = torch.tensor([enc_ids])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        ck, cv = enc(ids, mask)
        sk = torch.zeros(layers, 1, heads, 0, dk)
        sv = torch.zeros(layers, 1, heads, 0, dk)
        out = list(prefix)
        feed = torch.tensor([prefix])
        for _ in range(max_new):
            logits, sk, sv = dec(feed, mask, ck, cv, sk, sv)
            nxt = int(logits[0].argmax())
            out.append(nxt)
            if nxt == eos:
                break
            feed = torch.tensor([[nxt]])
    return out


class OrtGraphs:
    """The two exported graphs, decoded greedily the way the app does it."""

    def __init__(self, folder, layers, heads, head_dim, eos, max_new=MAX_NEW, threads=4):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enc = ort.InferenceSession(os.path.join(folder, "encoder.onnx"), opts, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(folder, "decoder.onnx"), opts, providers=["CPUExecutionProvider"])
        self.shape = (layers, 1, heads, 0, head_dim)
        self.eos, self.max_new = eos, max_new

    def greedy(self, enc_ids, prefix):
        ids = np.array([enc_ids], dtype=np.int64)
        mask = np.ones_like(ids)
        ck, cv = self.enc.run(None, {"input_ids": ids, "attention_mask": mask})
        sk = np.zeros(self.shape, np.float32)
        sv = np.zeros(self.shape, np.float32)
        out = list(prefix)
        feed = np.array([prefix], dtype=np.int64)
        for _ in range(self.max_new):
            logits, sk, sv = self.dec.run(None, {"input_ids": feed, "encoder_attention_mask": mask, "cross_k": ck, "cross_v": cv,
                                                 "self_k": sk, "self_v": sv})
            nxt = int(logits[0].argmax())
            out.append(nxt)
            if nxt == self.eos:
                break
            feed = np.array([[nxt]], dtype=np.int64)
        return out


class OrtModel(OrtGraphs):
    """The graphs with the tokenizer files next to them."""

    def __init__(self, folder, threads=4):
        self.spec = Spec(folder)
        m = self.spec.meta
        super().__init__(folder, m["layers"], m["heads"], m["headDim"], m["eos"], m["maxNew"], threads)

    def translate(self, text, src, tgt):
        return self.spec.decode(self.greedy(self.spec.encode(text, src, tgt), self.spec.decoder_prefix(tgt)))


def hf_generate(model, tok, family, enc_ids, tgt_token):
    from transformers import GenerationConfig
    cfg = GenerationConfig(num_beams=1, do_sample=False, max_new_tokens=MAX_NEW, decoder_start_token_id=model.config.decoder_start_token_id,
                           eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                           forced_bos_token_id=tok.convert_tokens_to_ids(tgt_token) if family == "m2m" else None)
    with torch.no_grad():
        return model.generate(torch.tensor([enc_ids]), generation_config=cfg)[0].tolist()


# ---------------------------------------------------------------------------------------------
# Export

def export(model_id, out):
    spec = MODELS[model_id]
    family = spec["family"]
    os.makedirs(out, exist_ok=True)
    work = out + ".fp32"
    os.makedirs(work, exist_ok=True)
    t0 = time.time()
    model, tok = load(model_id)
    print(f"loaded {spec['repo']} in {time.time() - t0:.0f} s, rss {rss_gb():.1f} GB")
    mismatches = tokenizer_files(model_id, model, tok, out)
    assert mismatches == 0, f"{mismatches} tokenizer mismatches"
    enc, dec = wrappers(model, family)
    d = dims(model, family)
    s = Spec(out)

    # Our step against transformers' generate(), on the sample sentences.
    differ = 0
    for src, tgt, text in SAMPLES:
        if src not in s.langs or tgt not in s.langs:
            continue
        enc_ids = s.encode(text, src, tgt)
        ref = hf_generate(model, tok, family, enc_ids, s.langs[tgt]["token"])
        ours = greedy_torch(enc, dec, enc_ids, s.decoder_prefix(tgt), s.meta["eos"], d["layers"], d["heads"], d["headDim"])
        same = ours == ref
        differ += not same
        print(f"{'same' if same else 'DIFFERENT'} {src}->{tgt}: {s.decode(ours)!r}" + ("" if same else f"\n  hf: {s.decode(ref)!r}"))
    assert differ == 0, f"our decoder step differs from transformers on {differ} sentences"

    export_graphs(enc, dec, d, s.encode("Hello there, how are you?", "en", "ru"), s.decoder_prefix("ru"), work)
    print(f"exported ({time.time() - t0:.0f} s, rss {rss_gb():.1f} GB)")
    del enc, dec, model
    gc.collect()

    for part in ["encoder", "decoder"]:
        quantize(f"{work}/{part}/{part}.onnx", f"{out}/{part}.onnx", spec["bits"])
        print(f"{part}: {os.path.getsize(f'{out}/{part}.onnx') / 1e6:.0f} MB ({time.time() - t0:.0f} s)")
    if os.environ.get("KEEP_FP32") != "1":
        shutil.rmtree(work)

    # The quantized graphs decode the samples (no exact match is required after quantization).
    m = OrtModel(out)
    for src, tgt, text in SAMPLES:
        if src in s.langs and tgt in s.langs:
            print(f"onnx {src}->{tgt}: {m.translate(text, src, tgt)!r}")
    json.dump({"revision": revision(model_id)}, open(f"{out}/source.json", "w"))


def export_graphs(enc, dec, d, enc_ids, dec_prefix, folder):
    """encoder.onnx and decoder.onnx (fp32; weights in files of their own past 2 GB) under folder/encoder, folder/decoder."""
    ids = torch.tensor([enc_ids])
    mask = torch.ones_like(ids)
    for part in ["encoder", "decoder"]:
        os.makedirs(f"{folder}/{part}", exist_ok=True)
    with torch.no_grad():
        ck, cv = enc(ids, mask)
        torch.onnx.export(enc, (ids, mask), f"{folder}/encoder/encoder.onnx",
                          input_names=["input_ids", "attention_mask"], output_names=["cross_k", "cross_v"],
                          dynamic_axes={"input_ids": {1: "src"}, "attention_mask": {1: "src"}, "cross_k": {3: "src"}, "cross_v": {3: "src"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)
        # Traced with a past of 3 positions and 2 new ones, so no size is fixed by the example.
        sk = torch.zeros(d["layers"], 1, d["heads"], 3, d["headDim"])
        torch.onnx.export(dec, (torch.tensor([dec_prefix]), mask, ck, cv, sk, torch.zeros_like(sk)), f"{folder}/decoder/decoder.onnx",
                          input_names=["input_ids", "encoder_attention_mask", "cross_k", "cross_v", "self_k", "self_v"],
                          output_names=["logits", "new_self_k", "new_self_v"],
                          dynamic_axes={"input_ids": {1: "tgt"}, "encoder_attention_mask": {1: "src"}, "cross_k": {3: "src"},
                                        "cross_v": {3: "src"}, "self_k": {3: "past"}, "self_v": {3: "past"},
                                        "new_self_k": {3: "total"}, "new_self_v": {3: "total"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)


def quantize(src, dst, bits):
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic
    if bits == 8:
        # MatMul weights and the vocabulary (Gather) to 8 bits; one shared matrix stays one copy.
        quantize_dynamic(src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"],
                         extra_options={"MatMulConstBOnly": True})
        return
    from onnx import version_converter
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    # 4-bit blocks for the weights and the vocabulary; int4 tensors need opset 21 (the export is 17).
    # Only the structure is converted (the weights stay in their files: a graph past 2 GB cannot be
    # serialized whole), saved next to them, and the quantizer loads it from there with the weights.
    model = version_converter.convert_version(onnx.load(src, load_external_data=False), 21)
    opset21 = os.path.join(os.path.dirname(os.path.abspath(src)), "opset21.onnx")
    onnx.save(model, opset21)
    del model
    q = MatMulNBitsQuantizer(onnx.load(opset21), block_size=32, is_symmetric=True, accuracy_level=4,
                             op_types_to_quantize=("MatMul", "Gather"), quant_axes=(("MatMul", 0), ("Gather", 1)))
    q.process()
    q.model.save_model_to_file(dst, use_external_data_format=False)
    os.remove(opset21)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


# ---------------------------------------------------------------------------------------------
# Quality and speed: FLORES-200 devtest, transformers (fp32, greedy) against the quantized graphs.

def check(model_id, out, flores):
    import sacrebleu
    spec = MODELS[model_id]
    family = spec["family"]
    s = Spec(out)
    directions = [(a, b) for a, b in DIRECTIONS if FLORES_TO_APP[a] in s.langs and FLORES_TO_APP[b] in s.langs]

    def lines(code):
        with open(os.path.join(flores, "devtest", f"{code}.devtest"), encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f][:FLORES_SENTENCES]

    cache = f"{out}.reference.json"
    if os.path.exists(cache):
        reference = json.load(open(cache))
    else:
        model, tok = load(model_id)
        reference, t0, n = {}, time.time(), 0
        for a, b in directions:
            src, tgt = FLORES_TO_APP[a], FLORES_TO_APP[b]
            reference[f"{a}>{b}"] = [s.decode(hf_generate(model, tok, family, s.encode(t, src, tgt), s.langs[tgt]["token"])) for t in lines(a)]
            n += FLORES_SENTENCES
            print(f"reference {a}->{b} done ({(time.time() - t0) / n:.2f} s/sentence)")
        del model
        gc.collect()
        json.dump(reference, open(cache, "w"), ensure_ascii=False)

    m = OrtModel(out)
    print(f"onnx sessions loaded, rss {rss_gb():.1f} GB")
    report = {"model": model_id, "sentences": FLORES_SENTENCES, "directions": []}
    worst = 0.0
    t_all, tok_all = 0.0, 0
    for a, b in directions:
        src, tgt = FLORES_TO_APP[a], FLORES_TO_APP[b]
        hyps, t0, tokens = [], time.time(), 0
        for t in lines(a):
            ids = m.greedy(s.encode(t, src, tgt), s.decoder_prefix(tgt))
            tokens += len(ids)
            hyps.append(s.decode(ids))
        dt = time.time() - t0
        t_all += dt
        tok_all += tokens
        refs = lines(b)
        ref_chrf = sacrebleu.corpus_chrf(reference[f"{a}>{b}"], [refs], word_order=2).score
        q_chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
        worst = max(worst, ref_chrf - q_chrf)
        row = {"direction": f"{src}->{tgt}", "chrf_fp32": round(ref_chrf, 1), "chrf_quantized": round(q_chrf, 1),
               "seconds_per_sentence": round(dt / len(hyps), 2), "example": hyps[0]}
        report["directions"].append(row)
        print(json.dumps(row, ensure_ascii=False))
    report["tokens_per_second"] = round(tok_all / t_all, 1)
    report["peak_rss_gb"] = round(rss_gb(), 2)
    report["worst_drop"] = round(worst, 2)
    json.dump(report, open(f"{out}/report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    assert worst <= 3.0, f"quantization costs {worst:.1f} chrF++ on some direction"


if __name__ == "__main__":
    cmd, model_id, out = sys.argv[1:4]
    if cmd == "export":
        export(model_id, out)
    elif cmd == "check":
        check(model_id, out, sys.argv[4])
    else:
        raise SystemExit(f"unknown command {cmd}")
