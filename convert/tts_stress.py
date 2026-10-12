#!/usr/bin/env python3
"""A Russian stress dictionary for the Silero voices, made from Vosk-TTS's pronunciation dictionary
(Apache-2.0, 2 million word forms): there the stressed vowel of each word is the phoneme marked 1, and
Russian vowel letters and vowel phonemes go one to one, so the stress becomes a vowel number.

Output, sorted by the UTF-8 bytes of the key (the app looks words up by binary search in place):

    <word, ё written as е>\\t<number of the stressed vowel, from 1>[ё]

"ё" after the number: the stressed vowel is "ё" (written "е" in most texts). Words of one vowel whose
"е" is not "ё" are left out (their stress is plain), as are words the dictionary spells differently
from how they read. Among homographs the more frequent reading wins.

    python3 tts_stress.py <vosk dictionary> <out.tsv>
"""
import sys

VOWEL_LETTERS = set("аеёиоуыэюя")
VOWEL_PHONEMES = {"a", "o", "u", "e", "i", "y"}
CYRILLIC = set("абвгдеёжзийклмнопрстуфхцчшщъыьэюя-")


def stress_of(word, phonemes):
    """(vowel number from 1, is it ё) or None when letters and phonemes do not line up."""
    vowels = [p for p in phonemes if p[:-1] in VOWEL_PHONEMES and p[-1] in "01"]
    letters = [c for c in word if c in VOWEL_LETTERS]
    if not vowels or len(vowels) != len(letters):
        return None
    stressed = [i for i, p in enumerate(vowels) if p.endswith("1")]
    if len(stressed) != 1:
        return None
    i = stressed[0]
    yo = letters[i] in "её" and vowels[i][:-1] == "o"
    return i + 1, yo


def main():
    source, target = sys.argv[1], sys.argv[2]
    best = {}
    for line in open(source, encoding="utf-8"):
        parts = line.split()
        if len(parts) < 3:
            continue
        word, prob, phonemes = parts[0], float(parts[1]), parts[2:]
        if not set(word) <= CYRILLIC or not any(c in VOWEL_LETTERS for c in word):
            continue
        stress = stress_of(word, phonemes)
        if stress is None:
            continue
        key = word.replace("ё", "е")
        if key not in best or prob > best[key][0]:
            best[key] = (prob, stress)
    rows = []
    for key, (_, (number, yo)) in best.items():
        single = sum(c in VOWEL_LETTERS for c in key) == 1
        if single and not yo:
            continue
        rows.append((key.encode("utf-8"), f"{key}\t{number}{'ё' if yo else ''}\n"))
    rows.sort()
    with open(target, "w", encoding="utf-8", newline="\n") as out:
        for _, row in rows:
            out.write(row)
    print(f"{len(rows)} words")


if __name__ == "__main__":
    main()
