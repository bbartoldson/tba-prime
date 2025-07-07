from __future__ import annotations

import argparse
import json
from pathlib import Path

import reasoning_gym
from datasets import Dataset

# Default output path (can be overridden via CLI).
OUTPUT_PATH = Path("/home/mila/j/jain.vineet/scratch/tba_prime/datasets/")

_PROMPT_TEMPLATE = (
    "You are given a problem, solve it step-by-step. "
    "Provide your final answer between <answer> and </answer> tags.\n\n"
    "Problem:\n{question}\n\nSolution:\n"
)


def build_dataset(task: str, size: int, seed: int) -> Dataset:
    """Generate *size* samples from *task* and return HF Dataset."""
    samples = reasoning_gym.create_dataset(task, size=size, seed=seed)  # type: ignore[attr-defined]

    converted = []
    for idx, entry in enumerate(samples):
        prompt = _PROMPT_TEMPLATE.format(question=entry["question"])

        verification_info = {
            "dataset_name": task,
            "entry": entry,
            "ground_truth": entry["answer"],
        }

        converted.append(
            {
                "problem_id": f"rg_{task}_{idx}",
                "task_type": "reasoning_gym",
                "prompt": prompt,
                "verification_info": json.dumps(verification_info),
            }
        )

    return Dataset.from_list(converted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Genesys dataset from Reasoning-Gym task")
    parser.add_argument("--task", type=str, default="mini_sudoku", help="Reasoning-Gym task name (e.g. mini_sudoku, arc_agi, …)")
    parser.add_argument("--size", type=int, default=1024, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for RG data generator")
    parser.add_argument("--out", type=str, default=str(OUTPUT_PATH), help="Path to save the dataset (directory will be created)")
    args = parser.parse_args()

    ds = build_dataset(args.task, args.size, args.seed)

    out_dir = Path(args.out) / f"{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(out_dir)
    print(f"Saved {len(ds):,} problems to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
