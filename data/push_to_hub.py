"""
Offline upload script — run after fetch_races.py completes.
Uploads all Parquet files from data/cache/ to HF Datasets
under the repository 'f1-race-logs'.

Not imported by app.py.

Usage:
    HF_TOKEN=hf_... python data/push_to_hub.py
    python data/push_to_hub.py --repo your-username/f1-race-logs
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo

CACHE_DIR = Path("data/cache")
DEFAULT_REPO = "f1-race-logs"


def push_to_hub(repo_id: str, token: str) -> None:
    api = HfApi(token=token)

    # Create dataset repo if it doesn't exist
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)

    parquet_files = sorted(CACHE_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {CACHE_DIR}. Run fetch_races.py first.")
        return

    print(f"Uploading {len(parquet_files)} Parquet files to {repo_id}...")

    for path in parquet_files:
        # Route laps vs weather into subdirectories
        subdir = "laps" if path.stem.endswith("_laps") else "weather"
        repo_path = f"{subdir}/{path.name}"

        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Add {path.name}",
        )
        print(f"  Uploaded {repo_path}")

    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="HF Dataset repo id, e.g. 'username/f1-race-logs'",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN environment variable before running.")

    push_to_hub(repo_id=args.repo, token=token)


if __name__ == "__main__":
    main()
