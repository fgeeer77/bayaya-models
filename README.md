# Bayaya models

Speech models for the [Bayaya](https://github.com/fgeeer77) voice-notes app that have **no official
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) build**, converted from their original sources.
Only files live here; the app downloads them from this repository's Releases, and only if the user
turns on "models from GitHub" in its settings.

| Release | Source | Licence |
|---|---|---|
| `whisper-ru-turbo-coriollon` | [coriollon/whisper-large-v3-turbo-russian](https://huggingface.co/coriollon/whisper-large-v3-turbo-russian) | Apache-2.0 |
| `whisper-ru-turbo-dvislobokov` | [dvislobokov/whisper-large-v3-turbo-russian](https://huggingface.co/dvislobokov/whisper-large-v3-turbo-russian) (the model behind MECHUK's whisper.cpp files) | MIT |
| `whisper-ru-large-v3-antony66` | [antony66/whisper-large-v3-russian](https://huggingface.co/antony66/whisper-large-v3-russian) (the model behind the popular CT2 builds) | not stated by the author (base Whisper: MIT) |
| `gigaam-multilingual-ctc` | [GigaAM Multilingual](https://github.com/salute-developers/GigaAM) `multilingual_ctc` (220M) | MIT |
| `gigaam-multilingual-large-ctc` | [GigaAM Multilingual](https://github.com/salute-developers/GigaAM) `multilingual_large_ctc` (600M) | MIT |

Voices (text to speech) for Bayaya Platinum, listed in `tts-catalog.json`:

| Release | Source | Licence |
|---|---|---|
| `tts-silero-v5-cis` | [Silero](https://github.com/snakers4/silero-models) `v5_cis_base_nostress`: 60 voices, Russian and 19 more languages of Russia and the CIS; Russian stress dictionary from the [Vosk-TTS](https://alphacephei.com/vosk/) ru 0.9 pronunciation dictionary | MIT (model), Apache-2.0 (dictionary) |
| `tts-kokoro-ru-base`, `tts-kokoro-ru-dima` | [zaakirio/kokoro-ru](https://huggingface.co/zaakirio/kokoro-ru) (ONNX by buchwandler/kokoro-onnx-models) | OpenRAIL-M (weights), Apache-2.0 (code) |
| `tts-kokoro-de-thorsten` | [Thorsten-Voice/Kokoro](https://huggingface.co/Thorsten-Voice/Kokoro) | Apache-2.0 |

## How a model gets here

Everything runs in [`convert.yml`](.github/workflows/convert.yml); nothing is uploaded by hand.

1. **Whisper**: the transformers checkpoint is converted to the openai-whisper format
   ([`whisper_hf_to_openai.py`](convert/whisper_hf_to_openai.py)) and exported by the **official
   sherpa-onnx script at release v1.13.8** with three asserted patches
   ([`export_whisper.py`](convert/export_whisper.py)), then quantised to int8.
   **GigaAM**: exported with GigaAM's own `to_onnx`, exactly like sherpa-onnx's official GigaAM v3 script
   ([`export_gigaam.py`](convert/export_gigaam.py)).
2. **Verification**: the converted model runs in sherpa-onnx (the runtime the app uses) on a test
   recording, and its transcript is compared with the **original** model's; the job fails if the
   character error rate between them exceeds 5 % (GigaAM) or 15 % (Whisper). The numbers are in each
   release's notes.
3. **Publishing**: each release lists the source revision, licence, verification output and the
   SHA-256 of every file (`SHA256SUMS`, `<id>.json`).

**Voices** ([`convert-tts.yml`](.github/workflows/convert-tts.yml)): Kokoro models are brought to the
sherpa-onnx layout ([`tts_kokoro.py`](convert/tts_kokoro.py)). Silero's TorchScript networks do not
export as they are, so [`tts_silero.py`](convert/tts_silero.py) rebuilds them as plain PyTorch modules
with the original weights, checks each against the original's part, exports two ONNX graphs and checks
the whole pipeline (the app's steps: durations, frame expansion, inverse STFT) against the original
model's audio (SNR at least 40 dB; 67–94 dB in practice). [`tts_stress.py`](convert/tts_stress.py)
turns Vosk-TTS's dictionary into a sorted stress table. Then every voice reads test sentences in the
app's runtime and a speech recogniser (GigaAM or Whisper) writes down what it heard; the release is
published only if the character error rate stays under the threshold.

## Catalog

`catalog.json` (speech recognition) and `tts-catalog.json` (voices, Bayaya Platinum only) describe the
models for the app: URLs, sizes and SHA-256. `catalog.json.sig` is its
RSA-SHA256 signature made with the app's own signing key; the app accepts the catalog only if the
signature verifies against the certificate it was installed with, and only if its `version` is not
older than one it has already seen. Every downloaded file must then match its SHA-256.
