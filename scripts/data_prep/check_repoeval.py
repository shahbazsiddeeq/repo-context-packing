from pathlib import Path

def main():
    benchmark_dir = Path("data/raw/repoeval/benchmark_files")
    repo_dir = Path("data/raw/repoeval/repositories")

    print("[RepoEval benchmark]")
    print("benchmark_dir exists:", benchmark_dir.exists())
    if benchmark_dir.exists():
        jsonl_files = sorted(benchmark_dir.glob("*.jsonl"))
        print("num jsonl files:", len(jsonl_files))
        for f in jsonl_files[:10]:
            print(" -", f.name)

    print("\n[RepoEval repositories]")
    print("repo_dir exists:", repo_dir.exists())
    if repo_dir.exists():
        for sub in sorted(repo_dir.iterdir()):
            if sub.is_dir():
                print(" -", sub.name)

if __name__ == "__main__":
    main()