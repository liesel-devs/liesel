"""Copy trusted CI notebooks into a staging directory or the docs source tree."""

import argparse
import shutil
from pathlib import Path


def copy_notebooks(executed, destination):
    notebooks = list(executed.rglob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No executed notebooks in {executed}")
    for notebook in notebooks:
        target = destination / notebook.relative_to(executed)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(notebook, target)
        # Keep one Sphinx source per page when replacing a Markdown guide.
        target.with_suffix(".md").unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executed", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    copy_notebooks(args.executed, args.destination)
