#!/usr/bin/env python3
"""Lists candidate translation models on Hugging Face: licence, gating, and the size of each large
file (ONNX exports, weights), to choose the big multilingual translator for Bayaya Platinum.

    python3 mt_survey.py
"""
import requests

MODELS = [
    "facebook/nllb-200-distilled-600M", "facebook/nllb-200-distilled-1.3B", "Xenova/nllb-200-distilled-600M",
    "facebook/m2m100_418M", "Xenova/m2m100_418M", "alirezamsh/small100",
    "google/madlad400-3b-mt", "jbochi/madlad400-3b-mt", "manancode/madlad400-3b-mt-onnx",
    "NiuTrans/LMT-60-0.6B", "NiuTrans/LMT-60-1.7B", "tencent/HY-MT1.5-1.8B", "tencent/Hy-MT1.5-1.8B-2bit-GGUF",
    "google/translategemma-4b-it", "litert-community/translategemma-4b-it", "CohereLabs/north-small-translate",
    "facebook/hf-seamless-m4t-medium", "Unbabel/Tower-Plus-2B",
]


def main():
    for m in MODELS:
        r = requests.get(f"https://huggingface.co/api/models/{m}", params={"blobs": "true"}, timeout=30)
        if r.status_code != 200:
            print(f"== {m}: HTTP {r.status_code}")
            continue
        d = r.json()
        lic = [t.removeprefix("license:") for t in d.get("tags", []) if t.startswith("license:")]
        print(f"== {m}  license={lic} gated={d.get('gated')} downloads={d.get('downloads')} lastModified={d.get('lastModified')}")
        for s in d.get("siblings", []):
            size = s.get("size") or 0
            if size > 30e6:
                print(f"   {size / 1e6:8.0f} MB  {s['rfilename']}")
    # Anything newer: search by keywords.
    for q in ["translation onnx", "nllb onnx", "madlad onnx", "translate litert"]:
        r = requests.get("https://huggingface.co/api/models", params={"search": q, "sort": "downloads", "limit": 15}, timeout=30)
        print(f"== search {q!r}: " + ", ".join(x["id"] for x in r.json()))


if __name__ == "__main__":
    main()
