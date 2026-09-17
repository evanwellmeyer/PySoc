import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HARNESS = ROOT / "reference" / "harness"


def require_exe(name):
    exe = HARNESS / name
    if not exe.exists():
        pytest.skip(f"reference executable {exe} not built")
    return exe
