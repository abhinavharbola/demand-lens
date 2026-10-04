import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_fixes_smoke_test_passes():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_notebook_fixes.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"notebook smoke test failed.\n\n{result.stdout}\n{result.stderr}"
    )
