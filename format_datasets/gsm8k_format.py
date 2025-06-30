#!/usr/bin/env python
# convert_gsm8k_to_genesys.py
import json, re, pathlib
from datasets import load_dataset, DatasetDict

root = pathlib.Path("/usr/workspace/venkatraman1/data/gsm8k-genesys")   # <- change if you like
root.mkdir(parents=True, exist_ok=True)

# --- helper ------------------------------------------------------------
extract = re.compile(r"####\s*([-+*/\d\s\.]+)")

def to_genesys(example):
    gold = extract.search(example["answer"])
    gold_num = gold.group(1).strip() if gold else example["answer"].strip()
    boxed = f"\\boxed{{{gold_num}}}"
    return {
        "prompt": example["question"].rstrip(),
        "verification_info": json.dumps({"ground_truth": boxed}),
        "task_type": "verifiable_math",
    }

# --- load GSM8K (≈8 k train, 1.3 k test) -------------------------------
raw = load_dataset("gsm8k", "main", split={"train":"train", "test":"test"})

ds = DatasetDict({k: v.map(to_genesys, remove_columns=v.column_names)
                  for k, v in raw.items()})

ds.save_to_disk(root)          # -> data/gsm8k-genesys/{train,test}
print("✓ Saved to", root.resolve())
