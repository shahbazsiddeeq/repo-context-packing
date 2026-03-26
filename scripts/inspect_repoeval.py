from pathlib import Path
import json

def main():
    benchmark_dir = Path("data/raw/repoeval/benchmark_files")
    if not benchmark_dir.exists():
        raise FileNotFoundError(f"Missing directory: {benchmark_dir}")

    files = sorted(benchmark_dir.glob("*.jsonl"))
    if not files:
        raise RuntimeError("No .jsonl files found in RepoEval benchmark_files")

    print("Available benchmark files:")
    for i, f in enumerate(files):
        print(f"[{i}] {f.name}")

    target = None
    for f in files:
        if "line" in f.name.lower():
            target = f
            break

    if target is None:
        target = files[0]

    print("\nSelected file:")
    print(target.name)

    with target.open("r", encoding="utf-8") as fh:
        first = json.loads(next(fh))

    print("\nTop-level keys:")
    print(sorted(first.keys()))

    print("\nFull first record:")
    print(json.dumps(first, indent=2)[:4000])

if __name__ == "__main__":
    main()