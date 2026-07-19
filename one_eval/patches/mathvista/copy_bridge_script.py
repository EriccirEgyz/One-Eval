"""
Helper script to copy MathVista bridge script to the cloned repository.
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

    # Source: patch file in One-Eval
    oneeval_root = Path(__file__).parent.parent.parent
    source = oneeval_root / "one_eval/patches/mathvista/run_mathvista_oneeval.py"

    # Destination: MathVista repo evaluation directory
    dest = repo_dir / "evaluation" / "run_mathvista_oneeval.py"

    if not source.exists():
        print(f"ERROR: Source file not found: {source}")
        sys.exit(1)

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)

    print(f"✓ Copied {source.name} to {dest}")
