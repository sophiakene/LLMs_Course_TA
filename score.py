"""Score saved predictions with the official cultural_robustness metric code.

    python score.py runs/baseline.jsonl runs/system_prompt.jsonl

Uses lm_eval.tasks.cultural_robustness.metrics from the LumiOpen fork, so the
numbers are computed the same way as in a normal lm-eval run. Run this as a
separate step after predict.py, so the embedding model gets the GPU to itself.

The embedding model is fixed for the course, so all scores are comparable.
The paper uses Qwen3-Embedding-8B, which does not fit on our T4 GPUs.

Writes, for each input file:
  <name>.scores.json        the summary scores
  <name>.per_question.csv   U and A for every question, for your analysis
"""

import argparse
import csv
import json
import os
from pathlib import Path

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"

# metrics.py reads the embedding model from this variable; overwrite whatever
# is set in the shell so every run uses the same judge.
os.environ["EMBEDDING_MODEL"] = EMBEDDING_MODEL

from lm_eval.tasks.cultural_robustness import metrics  # noqa: E402


def score_file(path: Path) -> tuple[dict, list[dict]]:
    metrics.reset_state()
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            metrics.record_response(
                {
                    "response": row["response"],
                    "base_id": row["base_id"],
                    "language": row["language"],
                    "type": row["type"],
                    "doc_id": row["id"],
                    "prompt": row["prompt"],
                }
            )

    # Same steps as metrics._scores(), but we keep the intermediate distances.
    unanchored = metrics._score_task("unspecific")  # U: per question + per language pair
    anchored = metrics._score_task("specific")      # A: averaged over anchors
    if unanchored is None or anchored is None:
        raise ValueError(f"{path}: need both prompt types answered in at least 2 languages")
    scores = {
        "cultural_diversity": sum(unanchored["question"].values()) / len(unanchored["question"]),
        "cultural_robustness": 1 - sum(anchored["question"].values()) / len(anchored["question"]),
    }
    csp = metrics._compute_csp(unanchored, anchored)
    scores.update(
        cultural_signal_purity=csp["csp"],
        cultural_signal_purity_question_level=csp["csp_question_level"],
        mean_pair_rho=csp["mean_pair_rho"],
        n_language_pairs=csp["n_pairs"],
        n_questions=csp["n_questions"],
        embedding_model=EMBEDDING_MODEL,
    )

    per_question = [
        {"question": q, "U": unanchored["question"][q], "A": anchored["question"].get(q)}
        for q in sorted(unanchored["question"], key=int)
    ]
    return scores, per_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", type=Path)
    args = parser.parse_args()

    results = {}
    for path in args.predictions:
        scores, per_question = score_file(path)
        results[path.stem] = scores
        path.with_suffix(".scores.json").write_text(json.dumps(scores, indent=2))
        with path.with_suffix(".per_question.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["question", "U", "A"])
            writer.writeheader()
            writer.writerows(per_question)

    # Scores are only comparable between runs on the same questions and setup.
    settings = {}
    for path in args.predictions:
        meta_path = path.with_suffix(".meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            settings[path.stem] = tuple(meta.get(k) for k in ("model", "languages", "split", "max_questions", "data"))
    if len(set(settings.values())) > 1:
        print("\nWARNING: these runs differ in model, languages, split or questions; do not compare them:")
        for name, values in settings.items():
            print(f"  {name}: model={values[0]} languages={values[1]} split={values[2]} max_questions={values[3]} data={values[4]}")

    # Comparison table
    columns =["cultural_signal_purity", "cultural_diversity", "cultural_robustness", "mean_pair_rho"]
    print(f"\n{'run':<25}" + "".join(f"{c[:22]:>24}" for c in columns))
    for name, scores in results.items():
        print(f"{name:<25}" + "".join(f"{scores[c]:>24.4f}" for c in columns))

    # TODO (analysis):
    # - Which questions have the highest U? Read those answers in runs/*.jsonl.
    #   Does a human agree they differ culturally?
    # - Which questions changed most between baseline and your solution, and why?
    # - Did your solution change U, A, or both? Was that your hypothesis?


if __name__ == "__main__":
    main()
