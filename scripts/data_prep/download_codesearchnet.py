from datasets import load_dataset
from pathlib import Path
import json

def save_jsonl(dataset_split, out_path, max_rows=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in dataset_split:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if max_rows is not None and count >= max_rows:
                break
    return count

def main():
    out_dir = Path("data/raw/codesearchnet")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading CodeSearchNet-derived Python benchmark from CodeXGLUE...")
    ds = load_dataset("google/code_x_glue_tc_nl_code_search_adv")

    for split in ["train", "validation", "test"]:
        split_ds = ds[split]

        sample_path = out_dir / f"python_{split}_sample.jsonl"
        n = save_jsonl(split_ds, sample_path, max_rows=100)

        print(f"Saved {n} rows to {sample_path}")

    print("Done.")

if __name__ == "__main__":
    main()