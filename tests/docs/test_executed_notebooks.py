import ast
import importlib.util
import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "docs/scripts/executed_notebooks.py"
SPEC = importlib.util.spec_from_file_location("executed_notebooks", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SCRIPT_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCRIPT_MODULE)


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def notebook(code, count=1, outputs=None, tags=()):
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# Demo\n"]},
            {
                "cell_type": "code",
                "execution_count": count,
                "metadata": {"tags": list(tags)},
                "outputs": outputs or [],
                "source": [code],
            },
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def make_repo(
    tmp_path,
    *,
    myst=False,
    code="answer = 42\n",
    tags=(),
    count=1,
    outputs=None,
):
    root = tmp_path / "repo"
    source = root / "docs/source"
    executed = root / "build/jupyter_execute"
    source.mkdir(parents=True)
    executed.mkdir(parents=True)
    native = notebook(code, count, outputs, tags)
    (source / "page.ipynb").write_text(json.dumps(notebook(code, None, tags=tags)))
    (source / "intro.md").write_text("Ordinary Markdown.\n")
    (executed / "page.ipynb").write_text(json.dumps(native))

    if myst:
        helper = source / "_examples/helper.py.inc"
        helper.parent.mkdir()
        helper.write_text("answer = 42\n")
        page = source / "tutorials/page.md"
        page.parent.mkdir()
        page.write_text(
            "---\nfile_format: mystnb\nkernelspec:\n"
            "  name: python3\n  display_name: Python 3\n---\n"
            "# Demo\n\n```{code-cell} ipython3\n"
            ":load: ../_examples/helper.py.inc\n```\n"
        )
        parsed = SCRIPT_MODULE.source_notebook(page)
        for cell in parsed["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = 1
                cell["outputs"] = []
        output = executed / "tutorials/page.ipynb"
        output.parent.mkdir()
        output.write_text(json.dumps(parsed))

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


def collect_bundle(root, executed, tmp_path):
    bundle = tmp_path / "bundle"
    result = cli(root, "collect", "--executed", str(executed), "--bundle", str(bundle))
    assert result.returncode == 0, result.stderr
    return bundle


def read_json(path):
    return json.loads(path.read_text())


def code_cell(notebook_data):
    return next(cell for cell in notebook_data["cells"] if cell["cell_type"] == "code")


def source_snapshot(root):
    source = root / "docs/source"
    return {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }


def make_remote(root, tmp_path):
    remote = tmp_path / "docs-outputs.git"
    subprocess.run(
        ["git", "init", "--bare", "-q", "-b", "main", str(remote)],
        check=True,
        capture_output=True,
        text=True,
    )
    git(root, "remote", "add", "origin", str(remote))
    return remote


def invoke_publish(root, bundle, monkeypatch, posts):
    event = root / "event.json"
    event.write_text("{}")
    for key, value in {
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_TOKEN": "github-test-token",
        "READTHEDOCS_TOKEN": "rtd-test-token",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(root)

    def request(request, timeout):
        posts.append(request.full_url)
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == "Token rtd-test-token"
        return nullcontext()

    monkeypatch.setattr(SCRIPT_MODULE, "urlopen", request)
    return SCRIPT_MODULE.main(["publish", "--bundle", str(bundle)])


def test_collect_apply_round_trip_has_only_commit_manifest(tmp_path):
    root, executed = make_repo(tmp_path, myst=True)
    commit = git(root, "rev-parse", "HEAD")
    bundle = collect_bundle(root, executed, tmp_path)

    assert read_json(bundle / "manifest.json") == {"format": 1, "commit": commit}
    assert {
        path.relative_to(bundle / "notebooks").as_posix()
        for path in (bundle / "notebooks").rglob("*.ipynb")
    } == {
        "page.ipynb",
        "tutorials/page.ipynb",
    }

    result = cli(root, "apply", "--bundle", str(bundle))

    assert result.returncode == 0, result.stderr
    assert read_json(root / "docs/source/page.ipynb") == read_json(
        bundle / "notebooks/page.ipynb"
    )
    applied = read_json(root / "docs/source/tutorials/page.ipynb")
    assert code_cell(applied)["source"] == "answer = 42\n"
    assert not (root / "docs/source/tutorials/page.md").exists()


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("commit", "commit does not match"),
        ("missing", "notebook set differs"),
        ("extra", "notebook set differs"),
        ("source", "cell source changed"),
        ("tags", "cell tags changed"),
        ("unexecuted", "unexecuted code cell"),
        ("error", "error output"),
    ],
)
def test_apply_refuses_invalid_bundle_without_changing_sources(
    tmp_path, corruption, message
):
    root, executed = make_repo(tmp_path, myst=True)
    bundle = collect_bundle(root, executed, tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = read_json(manifest_path)
    notebook_path = bundle / "notebooks/tutorials/page.ipynb"

    if corruption == "commit":
        manifest["commit"] = "0" * 40
        manifest_path.write_text(json.dumps(manifest))
    elif corruption == "missing":
        notebook_path.unlink()
    elif corruption == "extra":
        (bundle / "notebooks/extra.ipynb").write_text("{}")
    else:
        data = read_json(notebook_path)
        cell = code_cell(data)
        if corruption == "source":
            cell["source"] = ["different = True\n"]
        elif corruption == "tags":
            cell["metadata"]["tags"] = ["hide-input"]
        elif corruption == "unexecuted":
            cell["execution_count"] = None
            cell["outputs"] = []
        else:
            cell["outputs"] = [
                {
                    "output_type": "error",
                    "ename": "RuntimeError",
                    "evalue": "unexpected",
                    "traceback": ["RuntimeError: unexpected"],
                }
            ]
        notebook_path.write_text(json.dumps(data))

    before = source_snapshot(root)
    result = cli(root, "apply", "--bundle", str(bundle))

    assert result.returncode == 1
    assert message in result.stderr
    assert source_snapshot(root) == before


@pytest.mark.parametrize(
    "code,tags,count,outputs",
    [
        ("", (), None, []),
        ("answer = 42\n", ("skip-execution",), None, []),
        (
            "raise ValueError('expected')\n",
            ("raises-exception",),
            1,
            [
                {
                    "output_type": "error",
                    "ename": "ValueError",
                    "evalue": "expected",
                    "traceback": ["ValueError: expected"],
                }
            ],
        ),
    ],
)
def test_apply_accepts_nbclient_execution_cases(tmp_path, code, tags, count, outputs):
    root, executed = make_repo(
        tmp_path, code=code, tags=tags, count=count, outputs=outputs
    )
    bundle = collect_bundle(root, executed, tmp_path)

    result = cli(root, "apply", "--bundle", str(bundle))

    assert result.returncode == 0, result.stderr
    applied = read_json(root / "docs/source/page.ipynb")
    assert code_cell(applied)["execution_count"] == count
    assert code_cell(applied)["metadata"]["tags"] == list(tags)


def test_myst_config_matches_sphinx_settings():
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
    config = SCRIPT_MODULE.myst_config()
    assert config.enable_extensions == set(settings["myst_enable_extensions"])
    assert config.heading_anchors == settings["myst_heading_anchors"]
    assert config.dmath_double_inline == settings["myst_dmath_double_inline"]


def test_publish_first_bundle_wins(tmp_path, monkeypatch):
    root, executed = make_repo(tmp_path, myst=True)
    commit = git(root, "rev-parse", "HEAD")
    bundle = collect_bundle(root, executed, tmp_path)
    remote = make_remote(root, tmp_path)
    posts = []
    first = invoke_publish(root, bundle, monkeypatch, posts)
    stored = f"refs/heads/docs-outputs:outputs/{commit}/notebooks/page.ipynb"
    original = git(remote, "show", stored)
    changed = read_json(bundle / "notebooks/page.ipynb")
    code_cell(changed)["outputs"] = [
        {"output_type": "stream", "name": "stdout", "text": "later"}
    ]
    (bundle / "notebooks/page.ipynb").write_text(json.dumps(changed))
    second = invoke_publish(root, bundle, monkeypatch, posts)
    retained = git(remote, "show", stored)

    assert first == second == 0
    assert retained == original
    assert len(posts) == 2


def test_publish_refetches_and_retries_one_rejected_push(tmp_path, monkeypatch):
    root, executed = make_repo(tmp_path)
    bundle = collect_bundle(root, executed, tmp_path)
    remote = make_remote(root, tmp_path)
    rejected = remote / "first-push-rejected"
    hook = remote / "hooks/pre-receive"
    hook.write_text(
        "#!/bin/sh\n"
        'if [ ! -e "$(dirname "$0")/../first-push-rejected" ]; then\n'
        '  touch "$(dirname "$0")/../first-push-rejected"\n'
        "  exit 1\n"
        "fi\n"
        "exit 0\n"
    )
    hook.chmod(0o755)
    posts = []
    result = invoke_publish(root, bundle, monkeypatch, posts)

    assert result == 0
    assert rejected.exists()
    assert posts == [
        "https://app.readthedocs.org/api/v3/projects/liesel/versions/latest/builds/"
    ]
    assert git(remote, "rev-parse", "refs/heads/docs-outputs")
