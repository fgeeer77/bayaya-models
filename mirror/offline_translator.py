#!/usr/bin/env python3
"""Mirrors what Bayaya takes from Offline Translator's catalog (offline-translator.davidv.dev):

- Wiktionary dictionaries (tarkka format, data CC BY-SA from Wiktionary via Kaikki),
- the translation models Mozilla does not publish (Georgian and Swahili),

into one GitHub release, with a manifest of sizes and SHA-256 checksums. The app pins these files
and downloads them from here, so it does not depend on (or add load to) that project's CDN.

    python3 mirror/offline_translator.py out/
"""
import concurrent.futures
import hashlib
import json
import pathlib
import sys
import urllib.request

INDEX = "https://offline-translator.davidv.dev/index_v6.json"
EXTRA_TRANSLATION = ["translate-en-ka", "translate-ka-en", "translate-en-sw", "translate-sw-en"]


def fetch(url, attempts=4):
    last = None
    for _ in range(attempts):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "bayaya-mirror"})
            with urllib.request.urlopen(request, timeout=300) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
    raise last


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    index = json.loads(fetch(INDEX))
    packs = index["packs"]
    wanted = [pid for pid in packs if pid.startswith("dict-")] + EXTRA_TRANSLATION
    jobs = []
    for pid in wanted:
        for f in packs[pid]["files"]:
            url = f["url"]
            # Release asset names must be unique: the pack id prefixes each file.
            asset = f"{pid}--{url.rsplit('/', 1)[-1]}"
            jobs.append((pid, f, url, asset))

    def download(job):
        pid, f, url, asset = job
        data = fetch(url)
        (out / asset).write_bytes(data)
        return {
            "pack": pid,
            "role": f.get("role"),
            "name": f["name"],
            "asset": asset,
            "source": url,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "gzip": url.endswith(".gz"),
        }

    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        files = list(pool.map(download, jobs))
    manifest = {
        "source": INDEX,
        "generatedAt": index.get("generatedAt"),
        "dictionaryVersion": index.get("dictionaryVersion"),
        "packs": {pid: {k: v for k, v in packs[pid].items() if k != "files"} for pid in wanted},
        "files": files,
        "languages": {code: lang.get("meta", {}) for code, lang in index["languages"].items()},
    }
    (out / "mirror.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n")
    print(f"{len(files)} files, {sum(f['size'] for f in files) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
