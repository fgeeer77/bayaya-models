"""Downloads the PP-OCR ONNX models Bayaya Platinum pins (and any PP-OCRv6 ONNX PaddlePaddle publishes)."""
import json
import os
import sys
import urllib.request

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

# Repositories and revisions from Bayaya Platinum's model-pins.json.
PINNED = {
    "det": ("PaddlePaddle/PP-OCRv5_mobile_det_onnx", "e6f4fa85f00e168c862bc462aebca69eef9b3d3d"),
    "latin": ("PaddlePaddle/latin_PP-OCRv5_mobile_rec_onnx", "89d3a50e2c27e2e7cceeab0e944c25c807d5db4f"),
    "eslav": ("PaddlePaddle/eslav_PP-OCRv5_mobile_rec_onnx", "9a32171fc5718746875e1a261818884517975013"),
}


def get(url, path):
    with urllib.request.urlopen(url, timeout=300) as r, open(path, "wb") as f:
        f.write(r.read())
    print(f"{os.path.getsize(path):>10}  {os.path.basename(path)}", flush=True)


for key, (repo, rev) in PINNED.items():
    for name in ("inference.onnx", "inference.yml"):
        get(f"https://huggingface.co/{repo}/resolve/{rev}/{name}", os.path.join(OUT, f"v5_{key}_{name}"))

# PP-OCRv6: whatever PaddlePaddle has on Hugging Face (names, revisions, files).
found = {}
for query in ("PP-OCRv6", "OCRv6"):
    with urllib.request.urlopen(f"https://huggingface.co/api/models?author=PaddlePaddle&search={query}&limit=100", timeout=60) as r:
        for m in json.load(r):
            found[m["id"]] = m.get("sha")
print("PaddlePaddle v6 repos:", json.dumps(found, indent=1), flush=True)
for repo in sorted(found):
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}", timeout=60) as r:
        info = json.load(r)
    files = [s["rfilename"] for s in info.get("siblings", [])]
    print(repo, info.get("sha"), files, flush=True)
    wanted = ("tiny_det", "tiny_rec", "small_rec", "mobile_det", "mobile_rec")
    if repo.endswith("_onnx") and any(w in repo for w in wanted):
        tag = repo.split("/")[1].replace("_onnx", "")
        for name in files:
            if name.endswith((".onnx", ".yml", ".txt")):
                get(f"https://huggingface.co/{repo}/resolve/{info['sha']}/{name}", os.path.join(OUT, f"{tag}_{name.replace('/', '_')}"))
