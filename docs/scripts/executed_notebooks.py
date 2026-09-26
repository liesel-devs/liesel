"""Publish executed notebooks; validate their sources just before rendering.

Pages must embed their outputs. Generated external files are not published.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
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
        [
            "git",
            "-c",
            "user.name=github-actions[bot]",
            "-c",
            "user.email=github-actions[bot]@users.noreply.github.com",
            *args,
        ],
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


def gate(root, commit):
    tip = fetch_outputs(root)
    if tip is None or not git(root, "ls-tree", tip, f"outputs/{commit}").stdout:
        raise MissingBundle(f"no executed docs bundle for {commit}")


def apply_pinned(root):
    commit = git(root, "rev-parse", "HEAD").stdout.strip()
    tree = f"{PIN_REF}:outputs/{commit}"
    archive = subprocess.run(
        ["git", "archive", tree], cwd=root, check=True, capture_output=True, timeout=60
    ).stdout
    with tempfile.TemporaryDirectory() as temporary:
        bundle = Path(temporary)
        with tarfile.open(fileobj=io.BytesIO(archive)) as contents:
            contents.extractall(bundle, filter="data")
        apply(root, bundle)


def push_environment():
    environment = os.environ.copy()
    auth = base64.b64encode(
        f"x-access-token:{os.environ['GITHUB_TOKEN']}".encode()
    ).decode()
    environment.update(
        GIT_CONFIG_COUNT="1",
        GIT_CONFIG_KEY_0="http.extraheader",
        GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {auth}",
    )
    return environment


def publish(root, bundle):
    token = os.environ.get("READTHEDOCS_TOKEN")
    if not token:
        print("READTHEDOCS_TOKEN is empty; skipping publication")
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    kind = os.environ["GITHUB_EVENT_NAME"]
    if kind == "pull_request":
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
            git(work, "commit", "-qm", f"Publish docs {commit}")
            pushed = git(
                work,
                "push",
                f"--force-with-lease={OUTPUTS_REF}:{tip or ''}",
                "origin",
                f"HEAD:{OUTPUTS_REF}",
                check=False,
                env=push_environment(),
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


def keep_commits(root):
    refs = dict(
        line.split()[::-1]
        for line in git(
            root, "ls-remote", "origin", "refs/heads/main", "refs/tags/*"
        ).stdout.splitlines()
    )
    if "refs/heads/main" not in refs:
        raise BundleError("main is missing from the remote ref listing")
    kept = set(refs.values())
    repository = os.environ["GITHUB_REPOSITORY"]
    url = f"https://api.github.com/repos/{repository}/pulls?state=open&per_page=100"
    while url:
        request = Request(
            url, headers={"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
        )
        with urlopen(request, timeout=30) as response:
            pulls = json.loads(response.read())
            if not isinstance(pulls, list):
                raise BundleError("invalid pull request listing")
            kept.update(
                pull["head"]["sha"]
                for pull in pulls
                if (pull["head"]["repo"] or {}).get("full_name") == repository
            )
            next_page = re.search(
                r'<([^>]+)>; rel="next"', response.headers.get("Link", "")
            )
            url = next_page[1] if next_page else ""
    return kept


def prune(root, dry_run):
    tip = fetch_outputs(root)
    if tip is None:
        print("No docs outputs to prune")
        return
    kept = keep_commits(root)
    remote = git(root, "remote", "get-url", "origin").stdout.strip()
    with tempfile.TemporaryDirectory() as temporary:
        work = Path(temporary)
        git(work, "init", "-q")
        git(work, "fetch", "--depth=1", str(root), tip)
        git(work, "checkout", "--detach", "FETCH_HEAD")
        for folder in sorted((work / "outputs").glob("*")):
            retain = folder.name in kept
            print(f"{'keep' if retain else 'prune'} {folder.name}")
            if not retain:
                shutil.rmtree(folder)
        if dry_run:
            return
        git(work, "add", "-A")
        tree = git(work, "write-tree").stdout.strip()
        commit = git(
            work, "commit-tree", tree, "-m", "Prune docs outputs"
        ).stdout.strip()
        git(
            work,
            "push",
            f"--force-with-lease={OUTPUTS_REF}:{tip}",
            remote,
            f"{commit}:{OUTPUTS_REF}",
            env=push_environment(),
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    collector = commands.add_parser("collect")
    collector.add_argument("--executed", type=Path, required=True)
    collector.add_argument("--bundle", type=Path, required=True)
    applier = commands.add_parser("apply")
    applier.add_argument("--bundle", type=Path)
    gating = commands.add_parser("gate")
    gating.add_argument("--commit", required=True)
    publisher = commands.add_parser("publish")
    publisher.add_argument("--bundle", type=Path, required=True)
    pruner = commands.add_parser("prune")
    pruner.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd()
    try:
        if args.command == "collect":
            collect(root, args.executed, args.bundle)
        elif args.command == "gate":
            gate(root, args.commit)
        elif args.command == "publish":
            publish(root, args.bundle.resolve())
        elif args.command == "prune":
            prune(root, args.dry_run)
        elif args.bundle is None:
            apply_pinned(root)
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
