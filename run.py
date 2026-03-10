"""Launcher: sets PYTHONPATH and runs CIG CLI.
Usage: python run.py --seed "AI Ethics"
       python run.py --text "Raw text..."  |  python run.py --file path.txt
       python run.py --json for full JSON output. See README.md for data-gen and fine-tune."""

import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
python_dir = root / "python"
if str(python_dir) not in sys.path:
    sys.path.insert(0, str(python_dir))

os.chdir(root)

from goat_ts_cig.interface import main

if __name__ == "__main__":
    main()
