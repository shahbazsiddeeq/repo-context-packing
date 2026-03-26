from pathlib import Path

def main():
    required_dirs = [
        "data/raw",
        "data/processed",
        "src/chunking",
        "src/retrieval",
        "src/packing",
        "src/evaluation",
        "configs",
        "outputs/runs",
        "paper",
    ]

    missing = [d for d in required_dirs if not Path(d).exists()]

    if missing:
        print("Missing:", missing)
    else:
        print("Project structure looks good.")

if __name__ == "__main__":
    main()