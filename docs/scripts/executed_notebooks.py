"""Publish executed notebooks; validate their sources just before rendering.

Pages must embed their outputs. Generated external files are not published.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

OUTPUTS_REF = "refs/heads/docs-outputs"
PIN_REF = "refs/liesel/docs-outputs-pin"


class BundleError(Exception):
    pass


class MissingBundle(BundleError):
    pass


def git(root, *args, check=True, env=None):
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def myst_config():
    from myst_parser.config.main import MdParserConfig

    return MdParserConfig(
        enable_extensions={"amsmath", "dollarmath", "html_image"},
        heading_anchors=3,
        dmath_double_inline=True,
    )


def source_notebook(path):
    if path.suffix == ".ipynb":
        return json.loads(path.read_text())
    from myst_nb.core.read import read_myst_markdown_notebook

    return read_myst_markdown_notebook(
        path.read_text(), config=myst_config(), add_source_map=True, path=path
    )


def discover_pages(source):
    from myst_nb.core.read import is_myst_markdown_notebook

    return {
        path.relative_to(source).with_suffix(".ipynb"): path
        for path in sorted(source.rglob("*"))
        if ".ipynb_checkpoints" not in path.parts
        and (
            path.suffix == ".ipynb"
            or (path.suffix == ".md" and is_myst_markdown_notebook(path.read_text()))
        )
    }


def normalized_source(cell):
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def validate_cells(source, executed, path):
    before, after = source["cells"], executed["cells"]
    if len(before) != len(after):
        raise BundleError(f"cell count changed: {path}")
    for original, result in zip(before, after):
        if original["cell_type"] != result["cell_type"]:
            raise BundleError(f"cell type changed: {path}")
        if normalized_source(original) != normalized_source(result):
            raise BundleError(f"cell source changed: {path}")
        tags = original.get("metadata", {}).get("tags", [])
        if tags != result.get("metadata", {}).get("tags", []):
            raise BundleError(f"cell tags changed: {path}")
        if original["cell_type"] == "code":
            skipped = (
                not normalized_source(original).strip() or "skip-execution" in tags
            )
            if result.get("execution_count") is None and not skipped:
                raise BundleError(f"unexecuted code cell: {path}")
            if "raises-exception" not in tags and any(
                output["output_type"] == "error" for output in result.get("outputs", [])
            ):
                raise BundleError(f"error output: {path}")


def collect(root, executed, bundle):
    bundle.mkdir(parents=True)
    for notebook in executed.rglob("*.ipynb"):
        destination = bundle / "notebooks" / notebook.relative_to(executed)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(notebook, destination)
    manifest = {"format": 1, "commit": git(root, "rev-parse", "HEAD").stdout.strip()}
    (bundle / "manifest.json").write_text(json.dumps(manifest) + "\n")


def apply(root, bundle):
    manifest = json.loads((bundle / "manifest.json").read_text())
    if manifest.get("format") != 1:
        raise BundleError("unsupported bundle format")
    if manifest.get("commit") != git(root, "rev-parse", "HEAD").stdout.strip():
        raise BundleError("bundle commit does not match checkout")
    pages = discover_pages(root / "docs/source")
    notebooks = bundle / "notebooks"
    found = {path.relative_to(notebooks) for path in notebooks.rglob("*.ipynb")}
    if set(pages) != found:
        raise BundleError("bundle notebook set differs from executable pages")
    validated = []
    for relative, source in pages.items():
        content = (notebooks / relative).read_bytes()
        validate_cells(source_notebook(source), json.loads(content), source)
        validated.append((source, content))
    # Validate every page before changing any source.
    for source, content in validated:
        destination = source.with_suffix(".ipynb")
        destination.write_bytes(content)
        if destination != source:
            source.unlink()


def fetch_outputs(root):
    result = git(root, "ls-remote", "--exit-code", "origin", OUTPUTS_REF, check=False)
    if result.returncode == 2:
        return None
    result.check_returncode()
    git(root, "fetch", "--depth=1", "--no-tags", "origin", f"+{OUTPUTS_REF}:{PIN_REF}")
    return git(root, "rev-parse", PIN_REF).stdout.strip()


def gate(root, commit, pin):
    tip = fetch_outputs(root)
    if tip is None or not git(root, "ls-tree", tip, f"outputs/{commit}").stdout:
        raise MissingBundle(f"no executed docs bundle for {commit}")
    pin.parent.mkdir(parents=True, exist_ok=True)
    pin.write_text(json.dumps({"tree": f"{tip}:outputs/{commit}"}) + "\n")


def apply_pin(root, pin):
    tree = json.loads(pin.read_text())["tree"]
    archive = subprocess.run(
        ["git", "archive", tree], cwd=root, check=True, capture_output=True, timeout=60
    ).stdout
    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            contents.extractall(bundle, filter="data")
        apply(root, bundle)


def publish(root, bundle):
    token = os.environ.get("READTHEDOCS_TOKEN")
    if not token:
        print("READTHEDOCS_TOKEN is empty; skipping publication")
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    kind = os.environ["GITHUB_EVENT_NAME"]
    if kind == "pull_request":
        if (
            event["pull_request"]["head"]["repo"]["full_name"]
            != os.environ["GITHUB_REPOSITORY"]
        ):
            return
        slugs = [str(event["number"])]
    elif kind == "workflow_dispatch":
        slugs = [event["inputs"]["target"]]
    else:
        ref = os.environ["GITHUB_REF"]
        slugs = (
            [ref.removeprefix("refs/tags/"), "stable"]
            if ref.startswith("refs/tags/")
            else ["latest"]
        )
    commit = json.loads((bundle / "manifest.json").read_text())["commit"]
    remote = git(root, "remote", "get-url", "origin").stdout.strip()
    push_env = os.environ.copy()
    auth = base64.b64encode(
        f"x-access-token:{os.environ['GITHUB_TOKEN']}".encode()
    ).decode()
    push_env.update(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.extraheader",
        GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {auth}",
    )
    with tempfile.TemporaryDirectory() as temporary:
        for attempt in range(3):
            work = Path(temporary) / str(attempt)
            work.mkdir()
            git(work, "init", "-q")
            git(work, "remote", "add", "origin", remote)
            tip = fetch_outputs(work)
            if tip:
                git(work, "checkout", "--detach", tip)
            destination = work / "outputs" / commit
            if destination.exists():
                break
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bundle, destination)
            git(work, "add", "outputs")
            git(
                work,
                "-c",
                "user.name=Liesel docs",
                "-c",
                "user.email=docs@liesel.org",
                "commit",
                "-qm",
                f"Publish docs {commit}",
            )
            pushed = git(
                work,
                "push",
                f"--force-with-lease={OUTPUTS_REF}:{tip or ''}",
                "origin",
                f"HEAD:{OUTPUTS_REF}",
                check=False,
                env=push_env,
            )
            if pushed.returncode == 0:
                break
        else:
            pushed.check_returncode()
    for slug in slugs:
        request = Request(
            "https://app.readthedocs.org/api/v3/projects/liesel/versions/"
            f"{quote(slug, safe='')}/builds/",
            headers={"Authorization": f"Token {token}"},
            method="POST",
        )
        with urlopen(request, timeout=30):
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("collect")
    collector.add_argument("--executed", type=Path, required=True)
    collector.add_argument("--bundle", type=Path, required=True)
    applier = commands.add_parser("apply")
    inputs = applier.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bundle", type=Path)
    inputs.add_argument("--pin", type=Path)
    gating = commands.add_parser("gate")
    gating.add_argument("--commit", required=True)
    gating.add_argument("--pin", type=Path, required=True)
    publisher = commands.add_parser("publish")
    publisher.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path.cwd()
    try:
        if args.command == "collect":
            collect(root, args.executed, args.bundle)
        elif args.command == "gate":
            gate(root, args.commit, args.pin)
        elif args.command == "publish":
            publish(root, args.bundle.resolve())
        elif args.pin:
            apply_pin(root, args.pin)
        else:
            apply(root, args.bundle)
    except MissingBundle as exc:
        print(exc, file=sys.stderr)
        return 183
    except (
        BundleError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"executed-notebooks: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
