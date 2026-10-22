"""Load the cultural-robustness prompts.

Each row of LumiOpen/cultural-robustness looks like:

    {"id": "12",    "base_id": "12",    "language": "fi", "type": "unspecific", "prompt": "..."}
    {"id": "12-7",  "base_id": "12-7",  "language": "fi", "type": "specific",   "prompt": "... We live in X ..."}

- type "unspecific": unanchored question, no country given (101 per language)
- type "specific":   the same question anchored to a country/region; base_id is
                     "<question>-<anchor>" (many anchors per question)

The same base_id in different languages is the same prompt, translated. The
metric compares answers to the same base_id across languages.

The 101 questions are split into a development set (50 questions) and a test
set (51 questions), like in a shared task: develop on "dev", and run on "test"
only for your final evaluation.
"""

import random

from datasets import load_dataset


DATASET = "LumiOpen/cultural-robustness"

LANGUAGES = {
    "bn": "Bengali", "ca": "Catalan", "cs": "Czech", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "fi": "Finnish", "fo": "Faroese", "fr": "French", "he": "Hebrew",
    "hi": "Hindi", "it": "Italian", "kn": "Kannada", "mr": "Marathi",
    "pl": "Polish", "ru": "Russian", "sk": "Slovak", "sv": "Swedish",
    "ta": "Tamil", "te": "Telugu", "tr": "Turkish",
}


# Fixed for the course (a random draw with seed 2026), so everyone uses the same split.
TEST_QUESTIONS = {
    1, 2, 10, 11, 12, 13, 14, 15, 16, 19, 20, 24, 27, 29, 30, 31, 33, 37, 38, 41,
    42, 44, 45, 46, 49, 51, 52, 54, 57, 58, 60, 63, 65, 66, 67, 70, 71, 72, 74, 76,
    77, 78, 79, 80, 83, 86, 88, 93, 94, 98, 100,
}
SPLITS = ("dev", "test")


def question_id(doc: dict) -> str:
    """'12-7' -> '12': the question a prompt belongs to, ignoring the anchor."""
    return doc["base_id"].split("-")[0]


def in_split(doc: dict, split: str) -> bool:
    is_test = int(question_id(doc)) in TEST_QUESTIONS
    return is_test if split == "test" else not is_test


def load_prompts(
    languages: list[str],
    split: str,
    max_questions: int | None = None,
    seed: int = 0,
    path: str | None = None,
) -> list[dict]:
    """Return prompts for the given ISO language codes as a list of dicts.

    split is "dev" (50 questions, for development) or "test" (51 questions,
    for your final evaluation only).

    max_questions keeps a random subset of *questions*, with all their languages
    and anchors, so cross-language comparisons still work. Use it for quick smoke
    tests only. CSP needs at least 5 questions; use 20+ for anything you report.

    path loads your own prompts (a JSONL file with the same fields) instead of
    the benchmark; the split is then ignored.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, not {split!r}")
    unknown = set(languages) - set(LANGUAGES)
    if unknown:
        raise ValueError(f"Unknown language codes {sorted(unknown)}; choose from {sorted(LANGUAGES)}")

    if path:
        ds = load_dataset("json", data_files=path, split="train")
    else:
        ds = load_dataset(DATASET, split="train")
    wanted = set(languages)
    docs = [d for d in ds if d["language"] in wanted]
    if not path:
        docs = [d for d in docs if in_split(d, split)]

    if max_questions:
        questions = sorted({question_id(d) for d in docs}, key=int)
        keep = set(random.Random(seed).sample(questions, min(max_questions, len(questions))))
        docs = [d for d in docs if question_id(d) in keep]

    return docs


if __name__ == "__main__":
    # Quick look at the data: python data.py
    docs = load_prompts(["en", "de"], "dev", max_questions=3)
    print(f"{len(docs)} prompts")
    for d in docs[:3] + [d for d in docs if d["type"] == "specific"][:3]:
        print(d)
