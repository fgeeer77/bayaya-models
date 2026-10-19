#!/usr/bin/env python3
"""Checks the online translation services Bayaya Platinum can use: sends each the request the app
sends (app/.../translate/online/OnlineTranslators.kt) and prints what comes back, so a change on a
service's side shows up here before it does on a phone. Services that need a key are skipped.

    python3 online_probe.py
"""
import json
import random
import sys
import time
import urllib.parse
import uuid

import requests

TEXT = "The weather is nice today, let's go for a walk."
FROM, TO = "en", "ru"
TIMEOUT = 30


# Public servers of the engines whose address the user can change; the first one is the app's default.
LINGVA = ["https://lingva.ml", "https://lingva.lunar.icu", "https://translate.plausibility.cloud", "https://lingva.garudalinux.org"]
MOZHI = ["https://mozhi.pussthecat.org", "https://mozhi.aryak.me", "https://mo.zorby.top", "https://mozhi.canine.tools"]
SIMPLY = ["https://simplytranslate.org", "https://st.privacydev.net", "https://translate.northboot.xyz"]


def lingva(base):
    r = requests.get(f"{base}/api/v1/{FROM}/{TO}/{urllib.parse.quote(TEXT, safe='')}", timeout=TIMEOUT)
    return r, lambda j: j["translation"]


def mozhi(base, engine="google"):
    r = requests.post(f"{base}/api/translate/", files={k: (None, v) for k, v in
                      {"engine": engine, "from": "auto", "to": TO, "text": TEXT}.items()}, timeout=TIMEOUT)
    return r, lambda j: j["translated-text"]


def deepl_web():
    rid = random.randint(100_000, 199_999) * 1000 + 1
    count = TEXT.count("i") + 1
    now = int(time.time() * 1000)
    body = json.dumps({"jsonrpc": "2.0", "method": "LMT_handle_texts", "params": {
        "texts": [{"text": TEXT, "requestAlternatives": 0}], "splitting": "newlines",
        "lang": {"source_lang_user_selected": "auto", "target_lang": TO.upper()},
        "timestamp": now - now % count + count}, "id": rid}, separators=(",", ":"))
    body = body.replace('"method":"', '"method" : "' if (rid + 5) % 29 == 0 or (rid + 3) % 13 == 0 else '"method": "')
    r = requests.post("https://www2.deepl.com/jsonrpc", data=body.encode(), timeout=TIMEOUT, headers={
        "Content-Type": "application/json", "User-Agent": "DeepL-iOS/2.9.1 iOS 16.3.0 (iPhone13,2)", "x-app-os-name": "iOS",
        "x-app-os-version": "16.3.0", "x-app-device": "iPhone13,2", "x-app-build": "510265", "x-app-version": "2.9.1"})
    return r, lambda j: j["result"]["texts"][0]["text"]


def mymemory():
    r = requests.get("https://api.mymemory.translated.net/get", params={"q": TEXT, "langpair": f"{FROM}|{TO}"}, timeout=TIMEOUT)
    return r, lambda j: j["responseData"]["translatedText"]


def yandex():
    sid = uuid.uuid4().hex + "-0-0"
    r = requests.post("https://translate.yandex.net/api/v1/tr.json/translate",
                      params={"lang": f"{FROM}-{TO}", "srv": "android", "sid": sid}, data={"text": TEXT}, timeout=TIMEOUT)
    return r, lambda j: j["text"][0]


def simplytranslate(base, engine="google"):
    r = requests.get(f"{base}/api/translate/", params={"engine": engine, "from": "auto", "to": TO, "text": TEXT}, timeout=TIMEOUT)
    return r, lambda j: j["translated_text"] or (_ for _ in ()).throw(ValueError("empty translation"))


def mint():
    r = requests.post(f"https://translate.wmcloud.org/api/translate/{FROM}/{TO}", json={"text": TEXT}, timeout=TIMEOUT)
    return r, lambda j: j["translation"]


def glosbe():
    r = requests.post("https://translator-api.glosbe.com/translateByLangDetect", params={"sourceLang": FROM, "targetLang": TO},
                      data=TEXT.encode(), headers={"Content-Type": "text/plain; charset=utf-8"}, timeout=TIMEOUT)
    return r, lambda j: j["translation"]


def apertium():
    r = requests.post("https://apertium.org/apy/translate", files={k: (None, v) for k, v in
                      {"langpair": "eng|spa", "q": TEXT, "markUnknown": "no"}.items()}, timeout=TIMEOUT)
    return r, lambda j: j["responseData"]["translatedText"]


def pons():
    r = requests.post("https://api.pons.com/text-translation-web/v4/translate", params={"locale": "en"},
                      json={"sourceLanguage": FROM, "targetLanguage": TO, "text": TEXT}, timeout=TIMEOUT)
    return r, lambda j: j["text"]


def lara():
    r = requests.post("https://webapi.laratranslate.com/translate", json={"q": TEXT, "source": FROM, "target": TO}, timeout=TIMEOUT)
    return r, lambda j: j["content"]["translation"]


def libretranslate():
    r = requests.post("https://libretranslate.com/translate", json={"q": TEXT, "source": "auto", "target": TO, "format": "text"}, timeout=TIMEOUT)
    return r, lambda j: j["translatedText"]


def main():
    failed = 0
    probes = [(f"lingva {b}", lambda b=b: lingva(b)) for b in LINGVA]
    probes += [(f"mozhi {b} {e}", lambda b=b, e=e: mozhi(b, e)) for b in MOZHI for e in ["google", "yandex"]]
    probes += [(f"simply {b} {e}", lambda b=b, e=e: simplytranslate(b, e)) for b in SIMPLY for e in ["google", "reverso"]]
    probes += [(p.__name__, p) for p in [deepl_web, mymemory, yandex, mint, glosbe, apertium, pons, lara, libretranslate]]
    for name, probe in probes:
        try:
            response, pick = probe()
            try:
                text = pick(response.json())
                print(f"OK    {name:48} {response.status_code} {text!r}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name:48} {response.status_code} {type(e).__name__}: {response.text[:200]!r}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name:48} {type(e).__name__}: {str(e)[:200]}")
    print(f"{failed} failed (key-only services — DeepL API, Kagi, OneRing — not probed)")
    sys.exit(0)


if __name__ == "__main__":
    main()
