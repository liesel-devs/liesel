import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "docs/scripts/executed_notebooks.py"


def test_copy_notebooks_replaces_guides_and_preserves_other_files(tmp_path):
    executed = tmp_path / "executed"
    source = tmp_path / "source"
    (executed / "nested").mkdir(parents=True)
    (source / "nested").mkdir(parents=True)
    for name in ("nested/guide.ipynb", "native.ipynb"):
        (executed / name).write_bytes(b'{"cells": [], "metadata": {}}\n')
    (executed / "figure.png").write_bytes(b"not a notebook")
    (source / "nested/guide.md").write_text("Original Markdown guide")
    (source / "native.ipynb").write_text("Original notebook")
    (source / "index.md").write_text("Unrelated landing page")

    subprocess.run(
        [sys.executable, str(SCRIPT), str(executed), str(source)],
        check=True,
        timeout=30,
    )

    for name in ("nested/guide.ipynb", "native.ipynb"):
        assert (source / name).read_bytes() == (executed / name).read_bytes()
    assert not (source / "nested/guide.md").exists()
    assert not (source / "figure.png").exists()
    assert (source / "index.md").read_text() == "Unrelated landing page"
    assert (executed / "nested/guide.ipynb").exists()


def test_empty_notebook_directory_fails(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path), str(tmp_path / "source")],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "No executed notebooks" in result.stderr
    assert not (tmp_path / "source").exists()
