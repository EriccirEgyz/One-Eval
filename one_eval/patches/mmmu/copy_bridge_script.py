"""
Helper script to copy MMMU bridge script to the cloned repository.
Usage: python copy_bridge_script.py <repo_dir>
"""
import sys
import shutil
from pathlib import Path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python copy_bridge_script.py <repo_dir>")
        sys.exit(1)

    repo_dir = Path(sys.argv[1])

    oneeval_root = Path(__file__).parent.parent.parent
    source = oneeval_root / "one_eval/patches/mmmu/run_mmmu_oneeval.py"

    # Place in mmmu/ directory alongside main_parse_and_eval.py
    dest = repo_dir / "mmmu" / "run_mmmu_oneeval.py"

    if not source.exists():
        print(f"ERROR: Source file not found: {source}")
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)

    print(f"✓ Copied {source.name} to {dest}")
