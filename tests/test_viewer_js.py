"""Runs the viewer's JavaScript unit tests (tests/js) with Node's built-in test runner."""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_viewer_core_js():
    files = sorted(str(p) for p in (ROOT / "tests" / "js").glob("*.test.mjs"))
    assert files
    run = subprocess.run(["node", "--test", *files], capture_output=True, text=True, cwd=ROOT)
    assert run.returncode == 0, run.stdout + run.stderr
