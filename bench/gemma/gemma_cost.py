#!/usr/bin/env python3
"""
Processor time Bayaya Platinum spends on a forwarded voice message ("Распознай") with Gemma tasks on.

  asr <model dir> <flac>                  GigaAM v3 RNNT on the message, pieces <= 28 s, 2 and 4 threads
  gemma <model> <threads> <messages.json> Gemma 4 E2B: loading, then per message the app's own prompts in
                                          the app's order: clean-up, title and category, summary (200 chars)

CPU is process time (all threads), measured on the same runner for both, so they can be compared.
Temporary: part of a study, deleted with this branch.
"""
import json
import re
import sys
import time

SR = 16000
CATEGORIES = ["Идеи", "Задачи", "Встречи", "Работа", "Личное", "Покупки", "Здоровье", "Финансы"]
SUMMARY = ("Summarise this voice note in the language of the note, in at most 200 characters. "
           "Keep the key facts, decisions, names, dates and tasks. Output only the summary.")


def clock():
    return time.process_time(), time.time()


def since(start):
    cpu, wall = clock()
    return round(cpu - start[0], 2), round(wall - start[1], 2)


# The app's helpers (TextTasks): pieces of at most 2500 characters at sentence ends, excerpts of 6000.
def pieces(text, limit=2500):
    out, current = [], ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text):
        if current and len(current) + len(sentence) > limit:
            out.append(current)
            current = ""
        current = (current + " " + sentence).strip()
    if current:
        out.append(current)
    return out


def excerpt(text, limit=6000):
    return text if len(text) <= limit else text[:limit * 2 // 3] + "\n…\n" + text[-(limit // 3):]


def with_text(instruction, text):
    return f"{instruction}\n\n\"\"\"\n{text}\n\"\"\""


def describe_prompt():
    return ("You organise personal voice notes. Reply with ONLY a JSON object {\"title\": \"...\", \"category\": \"...\"}."
            "\n- title: 3 to 7 words in the language of the note that capture its main point; no quotes, no date."
            f"\n- category: exactly one of {', '.join(CATEGORIES)}.")


def text_of(response):
    content = response.get("content", "")
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return re.sub(r"<think>.*?</think>", "", str(content), flags=re.S).strip()


def asr(model_dir, flac):
    import numpy as np
    import sherpa_onnx
    import soundfile as sf

    cfg = "onnx-session.cfg"
    with open(cfg, "w") as f:
        f.write("SessionConfig.session.intra_op.allow_spinning=0\nSessionConfig.session.inter_op.allow_spinning=0\n")
    audio, rate = sf.read(flac, dtype="float32")
    assert rate == SR
    step = 28 * SR
    chunks = [audio[i:i + step] for i in range(0, len(audio), step)]
    for threads in (2, 4):
        start = clock()
        r = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=f"{model_dir}/encoder.int8.onnx", decoder=f"{model_dir}/decoder.onnx", joiner=f"{model_dir}/joiner.onnx",
            tokens=f"{model_dir}/tokens.txt", num_threads=threads, model_type="nemo_transducer", provider="cpu:" + cfg)
        load = since(start)
        start = clock()
        for c in chunks:
            s = r.create_stream()
            s.accept_waveform(SR, np.ascontiguousarray(c))
            r.decode_stream(s)
        run = since(start)
        print(json.dumps({"what": "asr", "threads": threads, "seconds": round(len(audio) / SR, 1),
                          "load_cpu": load[0], "load_wall": load[1], "cpu": run[0], "wall": run[1],
                          "cpu_per_audio_minute": round(run[0] * 60 * SR / len(audio), 2)}), flush=True)


def gemma(model, threads, messages_file):
    import litert_lm

    messages = json.load(open(messages_file, encoding="utf-8"))
    start = clock()
    engine = litert_lm.Engine(model, backend=litert_lm.Backend.CPU(thread_count=threads), cache_dir="cache")
    load = since(start)
    print(json.dumps({"what": "gemma-load", "threads": threads, "cpu": load[0], "wall": load[1]}), flush=True)

    def ask(prompt):
        conversation = engine.create_conversation(
            sampler_config=litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0),
            thinking_config=litert_lm.ThinkingConfig(enable_thinking=False))
        try:
            return text_of(conversation.send_message(prompt))
        finally:
            conversation.close()

    for n, m in enumerate(messages, 1):
        text = m["transcript"]
        start = clock()
        cleaned = " ".join(ask(with_text(
            "Below is an automatic speech-recognition transcript. Fix misrecognised words, punctuation, "
            "capitalisation and paragraphs. Do not add, remove, summarise or translate anything and keep the "
            "original language and wording. Output only the corrected text.", p)).strip('"') for p in pieces(text))
        cleanup = since(start)
        start = clock()
        description = ask(describe_prompt() + "\n\nTranscript:\n\"\"\"\n" + excerpt(cleaned) + "\n\"\"\"")
        describe = since(start)
        start = clock()
        summary = ask(with_text(SUMMARY, excerpt(cleaned)))
        summarise = since(start)
        print(json.dumps({"what": "gemma-tasks", "threads": threads, "message": n, "seconds": m["seconds"],
                          "chars_in": len(text), "chars_cleaned": len(cleaned),
                          "cleanup_cpu": cleanup[0], "cleanup_wall": cleanup[1],
                          "describe_cpu": describe[0], "describe_wall": describe[1],
                          "summary_cpu": summarise[0], "summary_wall": summarise[1],
                          "title": description, "summary": summary}, ensure_ascii=False), flush=True)
    engine.close()


if __name__ == "__main__":
    if sys.argv[1] == "asr":
        asr(sys.argv[2], sys.argv[3])
    else:
        gemma(sys.argv[2], int(sys.argv[3]), sys.argv[4])
