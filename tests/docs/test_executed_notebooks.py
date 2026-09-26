import ast
import base64
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docs/scripts/executed_notebooks.py"


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def notebook(code, execution_count=None, outputs=None):
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Demo\n"]},
            {
                "cell_type": "code",
                "execution_count": execution_count,
                "metadata": {},
                "outputs": outputs or [],
                "source": [code],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_repo(tmp_path):
    root = tmp_path / "repo"
    source = root / "docs/source"
    source.mkdir(parents=True)
    (root / "uv.lock").write_text("version = 1\n")
    (source / "page.ipynb").write_text(json.dumps(notebook("answer = 42\n")))
    (source / "intro.md").write_text("Ordinary Markdown.\n")
    (source / "not-a-notebook.md").write_text(
        "```{code-cell} ipython3\n"
        "print('not executable without MyST notebook metadata')\n"
        "```\n"
    )
    (source / "legacy.qmd").write_text(
        "---\nfile_format: mystnb\n---\n"
        "```{code-cell} ipython3\nprint('ignored')\n```\n"
    )
    checkpoint = source / ".ipynb_checkpoints"
    checkpoint.mkdir()
    (checkpoint / "page.ipynb").write_text(json.dumps(notebook("checkpoint = True\n")))
    executed = root / "build/jupyter_execute"
    executed.mkdir(parents=True)
    (executed / "page.ipynb").write_text(
        json.dumps(
            notebook(
                "answer = 42\n",
                1,
                [
                    {
                        "output_type": "execute_result",
                        "execution_count": 1,
                        "data": {"text/plain": "42"},
                        "metadata": {},
                    }
                ],
            )
        )
    )
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-qm", "fixture")
    return root, executed


def cli(root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def add_native_page(root, executed, name, code):
    source = root / "docs/source" / name
    target = executed / name
    source.parent.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(json.dumps(notebook(code)))
    target.write_text(json.dumps(notebook(code, 1)))
    git(root, "add", f"docs/source/{name}")
    git(root, "commit", "-qm", "add notebook fixture")


def add_native_execution_case(
    root, executed, name, code, tags, execution_count, outputs
):
    source = root / "docs/source" / name
    target = executed / name
    source_data = notebook(code)
    executed_data = notebook(code, execution_count, outputs)
    source_data["cells"][1]["metadata"]["tags"] = tags
    executed_data["cells"][1]["metadata"]["tags"] = tags
    source.write_text(json.dumps(source_data))
    target.write_text(json.dumps(executed_data))
    git(root, "add", f"docs/source/{name}")
    git(root, "commit", "-qm", "add notebook execution case")


def apply_native_execution_case(tmp_path, name, code, tags, execution_count, outputs):
    root, executed = make_repo(tmp_path)
    add_native_execution_case(
        root, executed, name, code, tags, execution_count, outputs
    )
    bundle = tmp_path / "bundle"
    collect_bundle(root, executed, bundle)
    result = cli(root, "apply", "--bundle", str(bundle))
    assert result.returncode == 0, result.stderr
    applied = json.loads((root / "docs/source" / name).read_text())
    cell = next(cell for cell in applied["cells"] if cell["cell_type"] == "code")
    assert cell["source"] == [code]
    assert cell["metadata"]["tags"] == tags
    assert cell["outputs"] == outputs
    return cell


def add_myst_page(root, executed):
    source_root = root / "docs/source"
    helper = source_root / "_examples/helper.py.inc"
    helper.parent.mkdir(parents=True)
    helper.write_text("answer = 42\n")
    page = source_root / "tutorials/page.md"
    page.parent.mkdir(parents=True)
    page.write_text(
        "---\nfile_format: mystnb\nkernelspec:\n"
        "  name: python3\n  display_name: Python 3\n---\n"
        "# Demo\n\n```{code-cell} ipython3\n"
        ":load: ../_examples/helper.py.inc\n```\n"
    )
    from myst_nb.core.read import read_myst_markdown_notebook
    from myst_parser.config.main import MdParserConfig

    source_nb = read_myst_markdown_notebook(
        page.read_text(), config=MdParserConfig(), add_source_map=True, path=page
    )
    executed_nb = json.loads(json.dumps(source_nb))
    for cell in executed_nb["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = 1
            cell["outputs"] = []
    output = executed / "tutorials/page.ipynb"
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps(executed_nb))
    git(
        root,
        "add",
        "docs/source/_examples/helper.py.inc",
        "docs/source/tutorials/page.md",
    )
    git(root, "commit", "-qm", "add MyST notebook fixture")


def collect_bundle(root, executed, bundle):
    result = cli(root, "collect", "--executed", str(executed), "--bundle", str(bundle))
    assert result.returncode == 0, result.stderr


def make_bundle(tmp_path):
    root, executed = make_repo(tmp_path)
    add_native_page(root, executed, "z.ipynb", "last = True\n")
    add_myst_page(root, executed)
    bundle = tmp_path / "bundle"
    collect_bundle(root, executed, bundle)
    return root, executed, bundle


def bundle_manifest(bundle):
    return json.loads((bundle / "manifest.json").read_text())


def write_manifest(bundle, manifest):
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def edit_bundled_notebook(bundle, manifest, source, edit):
    entry = next(row for row in manifest["notebooks"] if row["source"] == source)
    path = bundle / entry["notebook"]
    notebook_data = json.loads(path.read_text())
    edit(notebook_data)
    path.write_text(json.dumps(notebook_data))
    entry["notebook_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot(root):
    source = root / "docs/source"
    return {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


def test_collect_bundles_executed_native_notebooks_and_ignores_ordinary_markdown(
    tmp_path,
):
    root, executed = make_repo(tmp_path)
    bundle = tmp_path / "bundle"

    result = cli(root, "collect", "--executed", str(executed), "--bundle", str(bundle))

    assert result.returncode == 0, result.stderr
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["commit"] == git(root, "rev-parse", "HEAD")
    assert [page["source"] for page in manifest["notebooks"]] == ["page.ipynb"]
    assert (bundle / manifest["notebooks"][0]["notebook"]).read_text() == (
        executed / "page.ipynb"
    ).read_text()


def test_apply_round_trips_all_pages_after_validating_the_complete_bundle(tmp_path):
    root, _, bundle = make_bundle(tmp_path)

    result = cli(root, "apply", "--bundle", str(bundle))

    assert result.returncode == 0, result.stderr
    assert json.loads((root / "docs/source/page.ipynb").read_text()) == notebook(
        "answer = 42\n",
        1,
        [
            {
                "output_type": "execute_result",
                "execution_count": 1,
                "data": {"text/plain": "42"},
                "metadata": {},
            }
        ],
    )
    assert json.loads((root / "docs/source/z.ipynb").read_text()) == notebook(
        "last = True\n", 1
    )


def test_apply_accepts_blank_code_cells_without_execution_count(tmp_path):
    cell = apply_native_execution_case(tmp_path, "blank.ipynb", "", [], None, [])

    assert cell["execution_count"] is None


def test_apply_accepts_skip_execution_cells_without_execution_count(tmp_path):
    cell = apply_native_execution_case(
        tmp_path,
        "skipped.ipynb",
        "answer = 42\n",
        ["skip-execution"],
        None,
        [],
    )

    assert cell["execution_count"] is None


def test_apply_accepts_expected_errors_in_raises_exception_cells(tmp_path):
    outputs = [
        {
            "output_type": "error",
            "ename": "ValueError",
            "evalue": "expected",
            "traceback": ["ValueError: expected"],
        }
    ]
    cell = apply_native_execution_case(
        tmp_path,
        "expected-error.ipynb",
        "raise ValueError('expected')\n",
        ["raises-exception"],
        1,
        outputs,
    )

    assert cell["execution_count"] == 1


def test_myst_reader_config_matches_literal_sphinx_settings():
    names = {
        "myst_enable_extensions",
        "myst_heading_anchors",
        "myst_dmath_double_inline",
    }
    conf = ast.parse((SCRIPT.parents[1] / "source/conf.py").read_text())
    settings = {}
    for node in conf.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    settings[target.id] = ast.literal_eval(node.value)

    assert settings.keys() == names
    spec = importlib.util.spec_from_file_location("executed_notebooks", SCRIPT)
    assert spec is not None and spec.loader is not None
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    config = publisher.myst_config()

    assert config.enable_extensions == set(settings["myst_enable_extensions"])
    assert config.heading_anchors == settings["myst_heading_anchors"]
    assert config.dmath_double_inline == settings["myst_dmath_double_inline"]


def test_collect_expands_myst_load_and_apply_keeps_the_docname(tmp_path):
    root, _, bundle = make_bundle(tmp_path)
    source_root = root / "docs/source"
    page = source_root / "tutorials/page.md"
    result = cli(root, "apply", "--bundle", str(bundle))

    assert result.returncode == 0, result.stderr
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert "tutorials/page.md" in {entry["source"] for entry in manifest["notebooks"]}
    applied = json.loads((source_root / "tutorials/page.ipynb").read_text())
    assert (
        next(cell for cell in applied["cells"] if cell["cell_type"] == "code")["source"]
        == "answer = 42\n"
    )
    assert applied["metadata"]["source_map"] == [6, 9]
    assert not page.exists()


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("commit", "commit does not match"),
        ("lock", "uv.lock digest"),
        ("python", "Python minor version"),
        ("changed_source", "source digest"),
        ("changed_helper", "cell source changed"),
        ("changed_cell", "cell source changed"),
        ("changed_tags", "cell tags changed"),
        ("missing", "missing or unexpected files"),
        ("extra", "missing or unexpected files"),
        ("unexecuted", "unexecuted code cell"),
        ("error", "error output"),
        ("metadata", "notebook metadata changed"),
        ("attachments", "cell metadata or attachments changed"),
        ("path", "invalid relative path"),
        ("symlink", "symlink is not allowed"),
    ],
)
def test_apply_rejects_corrupt_bundle_without_changing_any_source(
    tmp_path, corruption, message
):
    root, _, bundle = make_bundle(tmp_path)
    manifest = bundle_manifest(bundle)
    bundle_arg = bundle

    if corruption == "commit":
        manifest["commit"] = "0" * 40
    elif corruption == "lock":
        (root / "uv.lock").write_text("changed lock\n")
    elif corruption == "python":
        manifest["python_minor"] = "0.0"
    elif corruption == "changed_source":
        source = root / "docs/source/z.ipynb"
        source.write_text(source.read_text().replace("last = True", "last = False"))
    elif corruption == "changed_helper":
        (root / "docs/source/_examples/helper.py.inc").write_text("answer = 41\n")
    elif corruption == "changed_cell":
        edit_bundled_notebook(
            bundle,
            manifest,
            "z.ipynb",
            lambda nb: next(
                cell for cell in nb["cells"] if cell["cell_type"] == "code"
            ).update(source=["last = False\n"]),
        )
    elif corruption == "changed_tags":
        edit_bundled_notebook(
            bundle,
            manifest,
            "tutorials/page.md",
            lambda nb: next(
                cell for cell in nb["cells"] if cell["cell_type"] == "code"
            )["metadata"].update(tags=["hide-input"]),
        )
    elif corruption == "attachments":
        edit_bundled_notebook(
            bundle,
            manifest,
            "z.ipynb",
            lambda nb: nb["cells"][0].update(
                attachments={"plot.png": {"image/png": ["aW1hZ2U="]}}
            ),
        )
    elif corruption in {"unexecuted", "error", "metadata"}:

        def edit(notebook_data):
            code = next(
                cell for cell in notebook_data["cells"] if cell["cell_type"] == "code"
            )
            if corruption == "unexecuted":
                code["execution_count"] = None
                code["outputs"] = []
            elif corruption == "error":
                code["outputs"] = [
                    {
                        "output_type": "error",
                        "ename": "RuntimeError",
                        "evalue": "bad",
                        "traceback": ["RuntimeError: bad"],
                    }
                ]
            else:
                notebook_data["metadata"]["unauthorized"] = "changed"

        edit_bundled_notebook(bundle, manifest, "z.ipynb", edit)
    elif corruption == "missing":
        (bundle / "notebooks/z.ipynb").unlink()
    elif corruption == "extra":
        (bundle / "notebooks/extra.ipynb").write_text("{}")
    elif corruption == "path":
        manifest["notebooks"][0]["source"] = "../outside.md"
    elif corruption == "symlink":
        target = tmp_path / "link-target"
        target.mkdir()
        link = tmp_path / "bundle-link"
        link.symlink_to(target, target_is_directory=True)
        bundle_arg = link / ".." / "bundle"

    write_manifest(bundle, manifest)
    before = source_snapshot(root)
    result = cli(root, "apply", "--bundle", str(bundle_arg))

    assert result.returncode == 1, result.stderr
    assert message in result.stderr
    assert source_snapshot(root) == before


def test_apply_records_but_does_not_enforce_runtime_platform_or_tool_versions(tmp_path):
    root, _, bundle = make_bundle(tmp_path)
    manifest = bundle_manifest(bundle)
    manifest["platform"] = "another platform"
    manifest["tool_versions"]["Sphinx"] = "different"
    write_manifest(bundle, manifest)

    result = cli(root, "apply", "--bundle", str(bundle))

    assert result.returncode == 0, result.stderr


def test_collect_rejects_unknown_generated_files_under_execution_output(tmp_path):
    root, executed = make_repo(tmp_path)
    (executed / "generated.csv").write_text("x,y\n1,2\n")
    bundle = tmp_path / "bundle"

    result = cli(root, "collect", "--executed", str(executed), "--bundle", str(bundle))

    assert result.returncode == 1
    assert "unsupported external assets" in result.stderr
    assert not bundle.exists()


def test_collect_allows_image_files_embedded_in_executed_output(tmp_path):
    root, executed = make_repo(tmp_path)
    image = b"small test image payload"
    encoded = base64.b64encode(image).decode("ascii")
    encoded = encoded[:8] + "\n" + encoded[8:]
    (executed / f"{hashlib.sha256(image).hexdigest()}.png").write_bytes(image)
    executed_notebook = json.loads((executed / "page.ipynb").read_text())
    code = next(
        cell for cell in executed_notebook["cells"] if cell["cell_type"] == "code"
    )
    code["outputs"] = [
        {
            "output_type": "display_data",
            "data": {"image/png": encoded},
            "metadata": {},
        }
    ]
    (executed / "page.ipynb").write_text(json.dumps(executed_notebook))
    bundle = tmp_path / "bundle"

    result = cli(root, "collect", "--executed", str(executed), "--bundle", str(bundle))

    assert result.returncode == 0, result.stderr
    assert (bundle / "notebooks/page.ipynb").is_file()
