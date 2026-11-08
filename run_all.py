import glob
import os
import subprocess
import sys

"""Run every pipeline script src/00 ... src/15 in order from the repo root and
stop at the first failure. Usage: python run_all.py
"""

ROOT = os.path.dirname(os.path.abspath(__file__))
scripts = sorted(glob.glob(os.path.join(ROOT, "src", "[0-9][0-9]_*.py")))

for path in scripts:
    name = os.path.relpath(path, ROOT)
    print(f"\n{'=' * 70}\nRunning {name}\n{'=' * 70}", flush=True)
    # cwd=ROOT because the scripts use paths like data/processed/...
    result = subprocess.run([sys.executable, path], cwd=ROOT)
    if result.returncode != 0:
        print(f"\nSTOPPED: {name} failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

print(f"\nAll {len(scripts)} scripts finished successfully.")
