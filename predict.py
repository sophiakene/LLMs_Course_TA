"""Generate answers with a solution and save them as JSONL.

    python predict.py --solution baseline --model Qwen/Qwen2.5-1.5B-Instruct \
        --languages en,de,fi --split dev --out runs/baseline_dev.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from data import SPLITS, load_prompts
from solutions import SOLUTIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True, choices=sorted(SOLUTIONS))
    parser.add_argument("--model", required=True, help="Hugging Face model id or local path")
    parser.add_argument("--languages", required=True, help="comma-separated ISO codes, e.g. en,de,fi")
    parser.add_argument("--split", required=True, choices=SPLITS, help="dev for development, test for the final evaluation")
    parser.add_argument("--max-questions", type=int, default=None, help="subsample questions for smoke tests")
    parser.add_argument("--data", default=None, help="your own prompts (JSONL) instead of the benchmark")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    docs = load_prompts(args.languages.split(","), args.split, args.max_questions, path=args.data)
    print(f"Loaded {len(docs)} prompts")

    solution = SOLUTIONS[args.solution](args.model)
    start = time.time()
    responses = solution.predict(docs)
    elapsed = time.time() - start
    assert len(responses) == len(docs), "predict() must return one answer per prompt"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for doc, response in zip(docs, responses):
            f.write(json.dumps({**doc, "response": response}, ensure_ascii=False) + "\n")
    # Save the settings next to the predictions so every run can be reproduced.
    meta = {**vars(args), "n_prompts": len(docs), "seconds": round(elapsed)}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out} ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
