"""Collect and apply notebooks executed by the documentation build."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

FORMAT = "liesel-executed-notebooks"
FORMAT_VERSION = 1
EXCLUDED_DIRS = {".ipynb_checkpoints"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMAGE_EXTENSIONS = {
    "image/png": {".png"},
    "image/jpeg": {".jpg", ".jpeg"},
    "image/gif": {".gif"},
    "image/svg+xml": {".svg"},
    "application/pdf": {".pdf"},
}


class BundleError(Exception):
    pass


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def safe_relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise BundleError(f"invalid relative path: {value!r}")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise BundleError(f"invalid relative path: {value!r}")
    if pure.as_posix() != value:
        raise BundleError(f"non-canonical relative path: {value!r}")
    return Path(*pure.parts)


def reject_symlinks(root: Path) -> None:
    reject_path_symlinks(root)
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            if path.is_symlink():
                raise BundleError(f"symlink is not allowed: {path}")


def reject_path_symlinks(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise BundleError(f"symlink is not allowed: {component}")


def myst_config():
    from myst_parser.config.main import MdParserConfig

    return MdParserConfig(
        enable_extensions={"amsmath", "dollarmath", "html_image"},
        heading_anchors=3,
        dmath_double_inline=True,
    )


def source_notebook(path: Path):
    if path.suffix == ".ipynb":
        try:
            import nbformat

            text = path.read_text()
            nbformat.reads(text, as_version=4)
            # Keep the source representation intact. nbformat may synthesize
            # cell IDs for older notebooks that do not yet have them.
            return json.loads(text)
        except Exception as exc:
            raise BundleError(f"cannot read notebook source {path}: {exc}") from exc

    from myst_nb.core.read import read_myst_markdown_notebook

    try:
        return read_myst_markdown_notebook(
            path.read_text(), config=myst_config(), add_source_map=True, path=path
        )
    except Exception as exc:
        raise BundleError(f"cannot parse MyST notebook source {path}: {exc}") from exc


def normalized_source(cell) -> str:
    value = cell.get("source", "")
    return "".join(value) if isinstance(value, list) else value


def validate_cells(source, executed, source_path: Path) -> None:
    source_cells = source.get("cells")
    executed_cells = executed.get("cells")
    if not isinstance(source_cells, list) or not isinstance(executed_cells, list):
        raise BundleError(f"invalid notebook cells: {source_path}")
    if len(source_cells) != len(executed_cells):
        raise BundleError(f"cell count changed: {source_path}")

    for index, (before, after) in enumerate(zip(source_cells, executed_cells)):
        if before.get("cell_type") != after.get("cell_type"):
            raise BundleError(f"cell type changed at {source_path}:{index}")
        if normalized_source(before) != normalized_source(after):
            raise BundleError(f"cell source changed at {source_path}:{index}")
        before_tags = before.get("metadata", {}).get("tags", [])
        after_tags = after.get("metadata", {}).get("tags", [])
        if before_tags != after_tags:
            raise BundleError(f"cell tags changed at {source_path}:{index}")
        if after.get("cell_type") == "code":
            if after.get("execution_count") is None:
                raise BundleError(f"unexecuted code cell at {source_path}:{index}")
            if any(
                output.get("output_type") == "error"
                for output in after.get("outputs", [])
            ):
                raise BundleError(f"error output at {source_path}:{index}")
        generated_id = source_path.suffix == ".md" or "id" not in before
        if comparable_cell(before, generated_id) != comparable_cell(
            after, generated_id
        ):
            raise BundleError(
                f"cell metadata or attachments changed at {source_path}:{index}"
            )


def comparable_cell(cell, generated_id: bool) -> dict:
    result = json.loads(json.dumps(cell))
    for key in ("source", "cell_type", "outputs", "execution_count"):
        result.pop(key, None)
    if generated_id:
        result.pop("id", None)
    result["metadata"] = comparable_metadata(result.get("metadata", {}))
    return result


def embedded_image_files(notebook) -> set[str]:
    files = set()
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            for mime, value in output.get("data", {}).items():
                if mime not in IMAGE_EXTENSIONS or not isinstance(value, (str, list)):
                    continue
                content = "".join(value) if isinstance(value, list) else value
                try:
                    data = (
                        os.linesep.join(content.splitlines()).encode("utf8")
                        if mime == "image/svg+xml"
                        else base64.b64decode(
                            re.sub(r"\s+", "", content), validate=True
                        )
                    )
                except (ValueError, TypeError) as exc:
                    raise BundleError(
                        f"invalid embedded {mime} notebook output: {exc}"
                    ) from exc
                files.update(
                    f"{digest(data)}{extension}" for extension in IMAGE_EXTENSIONS[mime]
                )
    return files


def discover_pages(source_root: Path) -> list[tuple[Path, str, str]]:
    if not source_root.is_dir():
        raise BundleError(f"documentation source directory is missing: {source_root}")
    reject_symlinks(source_root)
    pages = []
    docnames = set()

    for current, dirs, files in os.walk(source_root, followlinks=False):
        dirs[:] = sorted(name for name in dirs if name not in EXCLUDED_DIRS)
        for name in sorted(files):
            path = Path(current) / name
            if path.suffix not in {".md", ".ipynb"}:
                continue
            if path.suffix == ".md":
                from myst_nb.core.read import is_myst_markdown_notebook

                if not is_myst_markdown_notebook(path.read_text()):
                    continue
            relative = path.relative_to(source_root).as_posix()
            source_notebook(path)
            docname = PurePosixPath(relative).with_suffix("").as_posix()
            if docname in docnames:
                raise BundleError(f"duplicate notebook page: {docname}")
            docnames.add(docname)
            pages.append((path, relative, docname + ".ipynb"))

    return sorted(pages, key=lambda page: page[1])


def current_commit(root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise BundleError(f"cannot determine checked-out commit: {exc}") from exc


def runtime_versions() -> dict[str, str]:
    versions = {}
    for package in ("myst-nb", "myst-parser", "nbformat", "Sphinx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def collect(root: Path, executed_root: Path, bundle_root: Path) -> None:
    source_root = root / "docs/source"
    lockfile = root / "uv.lock"
    reject_path_symlinks(root)
    reject_path_symlinks(lockfile)
    if not lockfile.is_file():
        raise BundleError(f"lockfile is missing: {lockfile}")
    if bundle_root.exists() or bundle_root.is_symlink():
        raise BundleError(f"bundle destination already exists: {bundle_root}")

    pages = discover_pages(source_root)
    reject_symlinks(executed_root)
    expected = {notebook for _, _, notebook in pages}
    output_files = {
        path.relative_to(executed_root).as_posix()
        for path in executed_root.rglob("*")
        if path.is_file()
    }
    found = {path for path in output_files if path.endswith(".ipynb")}
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise BundleError(
            f"executed notebook set differs (missing={missing}, extra={extra})"
        )

    records = []
    embedded_images = set()
    bundle_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".executed-notebooks-", dir=bundle_root.parent
    ) as temp:
        stage = Path(temp)
        for source_path, source_rel, notebook_rel in pages:
            executed_path = executed_root / safe_relative(notebook_rel)
            if not executed_path.is_file():
                raise BundleError(f"executed notebook is missing: {executed_path}")
            source = source_notebook(source_path)
            try:
                executed = json.loads(executed_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise BundleError(
                    f"cannot read executed notebook {executed_path}: {exc}"
                ) from exc
            validate_cells(source, executed, source_path)
            embedded_images.update(embedded_image_files(executed))
            bundle_rel = "notebooks/" + notebook_rel
            bundle_file = stage / safe_relative(bundle_rel)
            bundle_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(executed_path, bundle_file)
            records.append(
                {
                    "source": source_rel,
                    "notebook": bundle_rel,
                    "source_sha256": file_digest(source_path),
                    "notebook_sha256": file_digest(bundle_file),
                }
            )

        unsupported_assets = (
            output_files
            - expected
            - {
                path
                for path in output_files
                if "/" not in path and path in embedded_images
            }
        )
        if unsupported_assets:
            raise BundleError(
                "unsupported external assets in execution output: "
                + ", ".join(sorted(unsupported_assets))
            )

        manifest = {
            "format": FORMAT,
            "version": FORMAT_VERSION,
            "commit": current_commit(root),
            "uv_lock_sha256": file_digest(lockfile),
            "python_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "tool_versions": runtime_versions(),
            "notebooks": records,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.rename(stage, bundle_root)


def comparable_metadata(metadata: dict, notebook_level=False) -> dict:
    result = json.loads(json.dumps(metadata))
    result.pop("execution", None)
    if notebook_level:
        result.pop("language_info", None)
        result.pop("widgets", None)
    return result


def validate_manifest(manifest, root: Path, bundle_root: Path, source_root: Path):
    fields = {
        "format",
        "version",
        "commit",
        "uv_lock_sha256",
        "python_minor",
        "python",
        "platform",
        "tool_versions",
        "notebooks",
    }
    if not isinstance(manifest, dict) or set(manifest) != fields:
        raise BundleError("manifest fields are invalid")
    if manifest["format"] != FORMAT or manifest["version"] != FORMAT_VERSION:
        raise BundleError("unsupported manifest format or version")
    if not isinstance(manifest["commit"], str) or manifest["commit"] != current_commit(
        root
    ):
        raise BundleError("bundle commit does not match the checked-out commit")
    lockfile = root / "uv.lock"
    reject_path_symlinks(lockfile)
    if not lockfile.is_file() or manifest["uv_lock_sha256"] != file_digest(lockfile):
        raise BundleError("uv.lock digest does not match the bundle")
    current_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    if manifest["python_minor"] != current_minor:
        raise BundleError("Python minor version does not match the bundle")
    if not SHA256.fullmatch(manifest["uv_lock_sha256"]):
        raise BundleError("manifest uv.lock digest is invalid")
    if not all(isinstance(manifest[key], str) for key in ("python", "platform")):
        raise BundleError("manifest runtime metadata is invalid")
    if not isinstance(manifest["tool_versions"], dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in manifest["tool_versions"].items()
    ):
        raise BundleError("manifest tool versions are invalid")

    pages = discover_pages(source_root)
    expected = {source: notebook for _, source, notebook in pages}
    entries = manifest["notebooks"]
    if not isinstance(entries, list):
        raise BundleError("manifest notebooks must be a list")
    by_source = {}
    notebooks = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "source",
            "notebook",
            "source_sha256",
            "notebook_sha256",
        }:
            raise BundleError("manifest notebook entry is invalid")
        source_rel = safe_relative(entry["source"]).as_posix()
        notebook_rel = safe_relative(entry["notebook"]).as_posix()
        if source_rel in by_source or notebook_rel in notebooks:
            raise BundleError("manifest paths must be unique")
        expected_notebook = expected.get(source_rel)
        if (
            expected_notebook is None
            or notebook_rel != "notebooks/" + expected_notebook
        ):
            raise BundleError(f"manifest contains an unexpected page: {source_rel}")
        if not all(
            isinstance(entry[key], str) and SHA256.fullmatch(entry[key])
            for key in ("source_sha256", "notebook_sha256")
        ):
            raise BundleError(f"manifest digest is invalid: {source_rel}")
        by_source[source_rel] = entry
        notebooks.add(notebook_rel)
    if set(by_source) != set(expected):
        missing = sorted(set(expected) - set(by_source))
        extra = sorted(set(by_source) - set(expected))
        raise BundleError(
            f"manifest page set differs (missing={missing}, extra={extra})"
        )

    actual_files = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    wanted_files = {"manifest.json", *notebooks}
    if actual_files != wanted_files:
        raise BundleError("bundle contains missing or unexpected files")

    validated = []
    for source_path, source_rel, notebook_rel in pages:
        entry = by_source[source_rel]
        bundle_notebook = bundle_root / safe_relative(entry["notebook"])
        reject_path_symlinks(source_path)
        reject_path_symlinks(bundle_notebook)
        if file_digest(source_path) != entry["source_sha256"]:
            raise BundleError(f"source digest does not match: {source_rel}")
        if file_digest(bundle_notebook) != entry["notebook_sha256"]:
            raise BundleError(f"notebook digest does not match: {source_rel}")
        source = source_notebook(source_path)
        try:
            import nbformat

            executed = nbformat.reads(bundle_notebook.read_text(), as_version=4)
            nbformat.validate(executed)
        except Exception as exc:
            raise BundleError(
                f"invalid executed notebook {bundle_notebook}: {exc}"
            ) from exc
        validate_cells(source, executed, source_path)
        if comparable_metadata(
            source.get("metadata", {}), notebook_level=True
        ) != comparable_metadata(executed.get("metadata", {}), notebook_level=True):
            raise BundleError(f"notebook metadata changed: {source_rel}")
        for index, (before, after) in enumerate(
            zip(source["cells"], executed["cells"])
        ):
            if comparable_metadata(before.get("metadata", {})) != comparable_metadata(
                after.get("metadata", {})
            ):
                raise BundleError(f"cell metadata changed at {source_rel}:{index}")
        destination = source_path.with_suffix(".ipynb")
        if destination != source_path and destination.exists():
            raise BundleError(f"notebook destination already exists: {destination}")
        reject_path_symlinks(destination)
        validated.append((source_path, destination, bundle_notebook.read_bytes()))
    return validated


def apply(root: Path, bundle_root: Path) -> None:
    reject_path_symlinks(root)
    reject_symlinks(bundle_root)
    if not bundle_root.is_dir():
        raise BundleError(f"bundle directory is missing: {bundle_root}")
    manifest_path = bundle_root / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read bundle manifest: {exc}") from exc
    source_root = root / "docs/source"
    validated = validate_manifest(manifest, root, bundle_root, source_root)

    # Stage every output before replacing or removing any source page.
    with tempfile.TemporaryDirectory(prefix=".apply-executed-", dir=root) as temp:
        stage = Path(temp)
        for index, (_, _, content) in enumerate(validated):
            (stage / f"{index}.ipynb").write_bytes(content)
        for index, (source, destination, _) in enumerate(validated):
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / f"{index}.ipynb", destination)
            if source != destination:
                source.unlink()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser(
        "collect", help="collect a fresh execution bundle"
    )
    collect_parser.add_argument("--executed", type=Path, required=True)
    collect_parser.add_argument("--bundle", type=Path, required=True)
    apply_parser = commands.add_parser(
        "apply", help="validate and apply an execution bundle"
    )
    apply_parser.add_argument("--bundle", type=Path, required=True)
    return result


def cli_path(path: Path, root: Path) -> Path:
    candidate = path if path.is_absolute() else root / path
    # Inspect the user-supplied path before abspath normalizes away ``..`` and
    # could hide a symlinked component.
    reject_path_symlinks(candidate)
    return candidate.absolute()


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    root = Path.cwd()
    try:
        if args.command == "collect":
            collect(root, cli_path(args.executed, root), cli_path(args.bundle, root))
        else:
            apply(root, cli_path(args.bundle, root))
    except BundleError as exc:
        print(f"executed-notebooks: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
