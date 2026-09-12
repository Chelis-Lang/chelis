#!/usr/bin/env python3
"""Author-time fragments and release-time assembly; contract: changelog.d/README.md."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .ci_detect_docs_only import is_doc_path
else:
    from ci_detect_docs_only import is_doc_path


REPO_ROOT = Path(__file__).resolve().parent.parent
CATEGORIES = ("added", "changed", "fixed")
FRAGMENT = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*\.(added|changed|fixed)(\.breaking)?\.md")
NUMBER = r"(?:0|[1-9][0-9]*)"
PRERELEASE = rf"(?:{NUMBER}|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER = re.compile(rf"{NUMBER}\.{NUMBER}\.{NUMBER}(?:-{PRERELEASE}(?:\.{PRERELEASE})*)?"
                    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?")
HEADER = re.compile(r"## \[([^\]]+)\](?: (?:-|\N{EM DASH}) ([0-9]{4}-[0-9]{2}-[0-9]{2}))?\s*")


class PolicyError(ValueError):
    """An authored input violates the documented changelog contract."""


@dataclass(frozen=True)
class File:
    mode: str
    data: bytes

    def text(self, path: str) -> str:
        if self.mode not in ("100644", "100755"):
            raise PolicyError(f"{path}: expected a regular file")
        try:
            return self.data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise PolicyError(f"{path}: expected UTF-8 text") from error


@dataclass(frozen=True)
class Fragment:
    path: str
    category: str
    breaking: bool
    body: str

    @property
    def content(self) -> tuple[str, bool, str]:
        return self.category, self.breaking, self.body


@dataclass(frozen=True)
class Section:
    version: str
    date: str
    start: int
    body_start: int
    end: int


@dataclass(frozen=True)
class Migration:
    changelog: bytes
    fragments: dict[str, bytes]


def version(value: str) -> str:
    if not SEMVER.fullmatch(value):
        raise PolicyError(f"invalid semantic version: {value!r}")
    return value


def release_date(value: str) -> str:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise PolicyError(f"invalid release date: {value!r}; expected YYYY-MM-DD")
    try:
        datetime.date.fromisoformat(value)
    except ValueError as error:
        raise PolicyError(f"invalid release date: {value!r}") from error
    return value


def required(tree: dict[str, File], path: str) -> str:
    if path not in tree:
        raise PolicyError(f"missing {path}")
    return tree[path].text(path)


def workspace_version(tree: dict[str, File]) -> str:
    try:
        value = tomllib.loads(required(tree, "Cargo.toml"))["workspace"]["package"]["version"]
        if not isinstance(value, str):
            raise PolicyError("Cargo.toml: workspace version must be a string")
        return version(value)
    except (tomllib.TOMLDecodeError, KeyError, TypeError) as error:
        raise PolicyError(f"Cargo.toml: invalid workspace.package.version: {error}") from error


def fragment(path: str, file: File) -> Fragment:
    text = file.text(path)
    name = path.removeprefix("changelog.d/")
    match = FRAGMENT.fullmatch(name)
    if match is None:
        raise PolicyError(f"{path}: expected <pr-or-slug>.<added|changed|fixed>[.breaking].md")
    body = text.replace("\r\n", "\n").strip()
    if not body:
        raise PolicyError(f"{path}: empty fragment")
    if re.match(r"(?:[-*+]\s|[0-9]+[.)]\s|#{1,6}\s|`{3}|~{3})", body):
        raise PolicyError(f"{path}: start with a paragraph, without an outer bullet or heading")
    fence = ""
    for line in body.splitlines():
        fence = fence_after(line, fence)
    if fence:
        raise PolicyError(f"{path}: unclosed code fence")
    return Fragment(path, match[1], bool(match[2]), body)


def fragments(tree: dict[str, File]) -> list[Fragment]:
    required(tree, "changelog.d/README.md")
    result = []
    for path, file in sorted(tree.items()):
        if path == "changelog.d" or (path.startswith("changelog.d/") and path != "changelog.d/README.md"):
            result.append(fragment(path, file))
    return result


def fence_after(line: str, fence: str) -> str:
    marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)$", line)
    if marker is None:
        return fence
    if fence:
        return "" if marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip() else fence
    # Backticks are forbidden in a backtick fence's info string by Markdown.
    return fence if marker[1][0] == "`" and "`" in marker[2] else marker[1]


def sections(text: str, *, allow_unreleased: bool = False) -> list[Section]:
    """Locate real release headings, ignoring fenced examples, without rewriting bytes."""
    headers = []
    fence = ""
    offset = 0
    seen = set()
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        was_fenced = bool(fence)
        fence = fence_after(stripped, fence)
        if not was_fenced and not fence and stripped.startswith("## ["):
            match = HEADER.fullmatch(stripped)
            if match is None:
                raise PolicyError(f"CHANGELOG.md: malformed release heading: {stripped!r}")
            if match[1] == "Unreleased":
                if not allow_unreleased:
                    raise PolicyError("CHANGELOG.md: move [Unreleased] notes into changelog.d/ before release assembly")
                if match[2] is not None:
                    raise PolicyError("CHANGELOG.md: legacy [Unreleased] heading must be undated")
                name = match[1]
            else:
                name = version(match[1])
            if name in seen:
                raise PolicyError(f"CHANGELOG.md: duplicate release section {name}")
            seen.add(name)
            if match[1] != "Unreleased" and match[2] is None:
                raise PolicyError(f"CHANGELOG.md: missing date for {name}")
            headers.append((name, "" if match[2] is None else release_date(match[2]), offset, offset + len(line)))
        offset += len(line)
    if fence:
        raise PolicyError("CHANGELOG.md: unclosed code fence")
    if not headers:
        raise PolicyError("CHANGELOG.md: no release sections")
    return [Section(*row, headers[i + 1][2] if i + 1 < len(headers) else len(text))
            for i, row in enumerate(headers)]


def migration_breaking(body: str) -> tuple[bool, str]:
    match = re.match(r"^\*\*BREAKING \(([^\r\n()*]+)\): ", body)
    if match is None:
        if re.match(r"^\*\*BREAKING\b", body):
            raise PolicyError("CHANGELOG.md: unsupported legacy BREAKING marker")
        return False, body
    return True, "**" + match[1] + ":" + body[match.end() - 1:]


def legacy_fragments(body: str) -> list[Fragment]:
    """Parse the deliberately small legacy-Unreleased grammar without guessing Markdown."""
    result = []
    category = None
    current: list[str] | None = None
    fence = ""

    def finish() -> None:
        nonlocal current, fence
        if current is None:
            return
        if fence:
            raise PolicyError("CHANGELOG.md: legacy [Unreleased] entry has an unclosed code fence")
        text = "\n".join(current).strip()
        breaking, text = migration_breaking(text)
        path = f"changelog.d/legacy-unreleased-{len(result) + 1:03d}.{category}"
        path += ".breaking.md" if breaking else ".md"
        result.append(fragment(path, File("100644", (text + "\n").encode())))
        current = None

    for raw in body.splitlines():
        if raw.startswith("### "):
            finish()
            name = raw.removeprefix("### ")
            if name.title() != name or name.lower() not in CATEGORIES:
                raise PolicyError(f"CHANGELOG.md: unsupported legacy [Unreleased] category {raw!r}")
            category = name.lower()
        elif raw.startswith("- "):
            if category is None:
                raise PolicyError("CHANGELOG.md: legacy [Unreleased] entry precedes a category")
            finish()
            current = [raw[2:]]
            fence = fence_after(current[0], fence)
        elif not raw.strip():
            if current is not None:
                current.append("")
        elif current is None:
            raise PolicyError("CHANGELOG.md: unsupported prose in legacy [Unreleased] section")
        elif not raw.startswith("  "):
            raise PolicyError("CHANGELOG.md: legacy [Unreleased] continuations require two-space indentation")
        else:
            line = raw[2:]
            current.append(line)
            fence = fence_after(line, fence)
    finish()
    if not result:
        raise PolicyError("CHANGELOG.md: legacy [Unreleased] section has no entries")
    return result


def migration(tree: dict[str, File]) -> Migration:
    workspace_version(tree)
    fragments(tree)
    text = required(tree, "CHANGELOG.md")
    found = sections(text, allow_unreleased=True)
    if found[0].version != "Unreleased":
        raise PolicyError("CHANGELOG.md: [Unreleased] must be the top release section for migration")
    if sum(section.version == "Unreleased" for section in found) != 1:
        raise PolicyError("CHANGELOG.md: expected exactly one top [Unreleased] section")
    if len(found) == 1:
        raise PolicyError("CHANGELOG.md: legacy [Unreleased] migration requires a historical release")
    legacy = found[0]
    notes = legacy_fragments(text[legacy.body_start:legacy.end])
    expected = {note.path: (note.body + "\n").encode() for note in notes}
    migrated = text[:legacy.start] + text[legacy.end:]
    sections(migrated)
    for path, data in expected.items():
        existing = tree.get(path)
        if existing is not None and existing != File("100644", data):
            raise PolicyError(f"{path}: migration fragment collision")
    return Migration(migrated.encode(), expected)


def render(notes: list[Fragment], requested: str, date: str) -> str:
    output = f"## [{requested}] - {date}\n\n"
    for category in CATEGORIES:
        group = sorted((n for n in notes if n.category == category), key=lambda n: (not n.breaking, n.path))
        if not group:
            continue
        output += f"### {category.title()}\n\n"
        for note in group:
            lines = note.body.splitlines()
            output += "- " + ("**BREAKING:** " if note.breaking else "") + lines[0] + "\n"
            output += "".join(("  " + line if line else "") + "\n" for line in lines[1:])
            output += "\n"
    return output


def assemble(tree: dict[str, File], requested: str, date: str) -> tuple[bytes, str, list[Fragment]]:
    version(requested)
    release_date(date)
    if requested != workspace_version(tree):
        raise PolicyError("release version must match Cargo.toml workspace.package.version")
    text = required(tree, "CHANGELOG.md")
    releases = sections(text)
    if any(s.version == requested for s in releases):
        raise PolicyError(f"release {requested} already exists")
    notes = fragments(tree)
    if not notes:
        raise PolicyError("no fragments to assemble")
    new_section = render(notes, requested, date)
    at = releases[0].start
    assembled = text[:at] + new_section + text[at:]
    if len(sections(assembled)) != len(releases) + 1:
        raise PolicyError("fragment Markdown obscures a release heading; check code fences")
    return assembled.encode("utf-8"), new_section, notes


def extract(tree: dict[str, File], requested: str) -> str:
    version(requested)
    if requested != workspace_version(tree):
        raise PolicyError("release version must match Cargo.toml workspace.package.version")
    if fragments(tree):
        raise PolicyError("unconsumed fragments remain in changelog.d/")
    text = required(tree, "CHANGELOG.md")
    matches = [s for s in sections(text) if s.version == requested]
    if not matches:
        raise PolicyError(f"missing release section {requested}")
    section = matches[0]
    body = text[section.body_start:section.end]
    if not body.strip() or not any(line.strip() and not line.lstrip().startswith("#") for line in body.splitlines()):
        raise PolicyError(f"empty release section {requested}")
    return text[section.start:section.end]


def disk_tree(root: Path) -> dict[str, File]:
    paths = [root / "CHANGELOG.md", root / "Cargo.toml"]
    directory = root / "changelog.d"
    if directory.is_symlink() or (directory.exists() and not directory.is_dir()):
        paths.append(directory)
    elif directory.exists():
        paths.extend(directory.iterdir())
    tree = {}
    for path in paths:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            continue
        regular = stat.S_ISREG(mode)
        tree[path.relative_to(root).as_posix()] = File(
            "100755" if regular and mode & stat.S_IXUSR else "100644" if regular else "other",
            path.read_bytes() if regular else b"",
        )
    return tree


def atomic_write(path: Path, data: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(stat.S_IMODE(path.stat().st_mode) if path.exists() else 0o644)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_new_file(path: Path, data: bytes) -> None:
    """Create a migration fragment without ever replacing an existing path."""
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o644)


def git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True)
    if result.returncode:
        raise OSError(f"git {' '.join(args)} failed: {result.stderr.decode('utf-8', errors='replace').strip()}")
    return result.stdout


def git_tree(root: Path, commit: str) -> dict[str, File]:
    tree = {}
    for row in git(root, "ls-tree", "-rz", commit, "--", "CHANGELOG.md", "Cargo.toml", "changelog.d").split(b"\0"):
        if not row:
            continue
        metadata, raw_path = row.split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split()
        path = raw_path.decode("utf-8")
        tree[path] = File(mode, git(root, "cat-file", "blob", oid) if kind == "blob" else b"")
    return tree


def needs_fragment(path: str) -> bool:
    normative = (bool(re.fullmatch(r"spec/[0-9]{2}-[^/]+\.md", path))
                 or path.startswith("spec/registry/")
                 or (path.startswith("openspec/") and "/specs/" in path and path.endswith("/spec.md")))
    return normative or not is_doc_path(path)


def is_release(before: dict[str, File], after: dict[str, File]) -> bool:
    try:
        current = workspace_version(after)
        if current == workspace_version(before) or fragments(after):
            return False
        first = sections(required(after, "CHANGELOG.md"))[0]
        if first.version != current:
            return False
        source = dict(before, **{"Cargo.toml": after["Cargo.toml"]})
        expected, _, _ = assemble(source, current, first.date)
        return expected == after["CHANGELOG.md"].data
    except PolicyError:
        return False


def is_unreleased_migration(before: dict[str, File], after: dict[str, File]) -> bool:
    """Admit only exact conservation of the one supported legacy representation."""
    try:
        if workspace_version(before) != workspace_version(after):
            return False
        expected = migration(before)
        if after.get("CHANGELOG.md") != File(before["CHANGELOG.md"].mode, expected.changelog):
            return False
        for path, file in before.items():
            if path.startswith("changelog.d/") and path != "changelog.d/README.md" and after.get(path) != file:
                return False
        return all(after.get(path) == File("100644", data) for path, data in expected.fragments.items())
    except PolicyError:
        return False


def check_pr(root: Path, base: str, head: str, labels: set[str]) -> list[str]:
    base = git(root, "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}").decode().strip()
    head = git(root, "rev-parse", "--verify", "--end-of-options", f"{head}^{{commit}}").decode().strip()
    ancestor = git(root, "merge-base", base, head).decode().strip()
    paths = [p.decode("utf-8") for p in git(root, "diff", "--no-renames", "--name-only", "-z", ancestor, head, "--").split(b"\0") if p]
    before, after = git_tree(root, ancestor), git_tree(root, head)
    findings = []
    validators = [fragments]
    if "CHANGELOG.md" in paths:
        validators.append(lambda t: sections(required(t, "CHANGELOG.md")))
    for validate in validators:
        try:
            validate(after)
        except PolicyError as error:
            findings.append(str(error))
    release = is_release(before, after)
    legacy_migration = is_unreleased_migration(before, after)
    if "CHANGELOG.md" in paths and not release and not legacy_migration:
        findings.append("direct CHANGELOG.md edit: expected a reproducible release assembly with unchanged history")
    old_content = set()
    for path, file in before.items():
        if path.startswith("changelog.d/") and path != "changelog.d/README.md":
            try:
                old_content.add(fragment(path, file).content)
            except PolicyError:
                pass  # A correction to an invalid old fragment is still an authored note.
    if legacy_migration:
        # Moving existing prose is not newly authored release content, even
        # when the same entry is also copied to another fragment filename.
        old_content.update(fragment(path, File("100644", data)).content
                           for path, data in migration(before).fragments.items())
    added_note = False
    for path in paths:
        if path.startswith("changelog.d/") and path != "changelog.d/README.md" and path in after:
            try:
                added_note |= fragment(path, after[path]).content not in old_content
            except PolicyError:
                pass  # The validation above reports malformed current fragments.
    if not release and not added_note and "no-changelog" not in labels and any(needs_fragment(p) for p in paths):
        findings.append("missing fragment: add or substantively amend a changelog.d/ entry, or apply no-changelog for internal work")
    return findings


def event_labels(path: str | None) -> set[str]:
    if path is None:
        return set()
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        labels = payload["pull_request"]["labels"]
        if not isinstance(labels, list) or any(not isinstance(row, dict) or not isinstance(row.get("name"), str) for row in labels):
            raise ValueError("expected pull_request.labels with string names")
        return {row["name"] for row in labels}
    except (ValueError, KeyError, TypeError) as error:
        raise OSError(f"cannot read PR labels from {path}: {error}") from error


def main(argv: list[str] | None = None, *, root: Path = REPO_ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="validate without writing")
    build = commands.add_parser("build", help="preview a release; --write consumes fragments")
    build.add_argument("--version", required=True)
    build.add_argument("--date", required=True)
    build.add_argument("--write", action="store_true")
    migrate = commands.add_parser("migrate-unreleased", help="preview lossless migration of the legacy top [Unreleased] section")
    migrate.add_argument("--write", action="store_true")
    extraction = commands.add_parser("extract", help="extract the tagged release's committed notes")
    extraction.add_argument("--version", required=True)
    extraction.add_argument("--output", type=Path, required=True)
    pr = commands.add_parser("check-pr", help="check a PR's merge-base diff")
    pr.add_argument("--base", required=True)
    pr.add_argument("--head", required=True)
    pr.add_argument("--event-file", default=os.environ.get("GITHUB_EVENT_PATH"))
    args = parser.parse_args(argv)
    try:
        if args.command == "check-pr":
            findings = check_pr(root, args.base, args.head, event_labels(args.event_file))
            for finding in findings:
                escaped = finding.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
                print(f"::error::{escaped}")
            message = "Changelog: FAIL" if findings else "Changelog: PASS"
            print(message)
            if summary := os.environ.get("GITHUB_STEP_SUMMARY"):
                with Path(summary).open("a", encoding="utf-8") as stream:
                    stream.write(message + "\n\n" + "".join(f"- {finding}\n" for finding in findings))
            return 1 if findings else 0
        tree = disk_tree(root)
        if args.command == "check":
            notes = fragments(tree)
            sections(required(tree, "CHANGELOG.md"))
            print(f"Changelog: PASS ({len(notes)} pending fragments)")
        elif args.command == "build":
            data, section, notes = assemble(tree, args.version, args.date)
            print(section, end="")
            print("Consume:\n" + "\n".join(n.path for n in notes), file=sys.stderr)
            if args.write:
                if disk_tree(root) != tree:
                    raise OSError("changelog inputs changed during assembly; retry the preview")
                atomic_write(root / "CHANGELOG.md", data)
                for note in notes:
                    path = root / note.path
                    if path.is_symlink() or path.read_bytes() != tree[note.path].data:
                        raise OSError(f"{note.path} changed during assembly; not deleting it")
                    path.unlink()
        elif args.command == "migrate-unreleased":
            plan = migration(tree)
            for path in sorted(plan.fragments):
                print(path)
            if args.write:
                expected_tree = dict(tree)
                expected_tree.update({path: File("100644", data) for path, data in plan.fragments.items()})
                if disk_tree(root) != tree:
                    raise OSError("changelog inputs changed during migration; retry the preview")
                for name, data in plan.fragments.items():
                    path = root / name
                    if name not in tree:
                        write_new_file(path, data)
                if disk_tree(root) != expected_tree:
                    raise OSError("changelog inputs changed during migration; legacy section remains")
                atomic_write(root / "CHANGELOG.md", plan.changelog)
        else:
            output = args.output.resolve()
            if output in (root / "CHANGELOG.md", root / "Cargo.toml") or output.is_relative_to(root / "changelog.d"):
                raise PolicyError("extraction output must not overwrite changelog inputs")
            text = extract(tree, args.version.removeprefix("v"))
            atomic_write(output, text.encode("utf-8"))
        return 0
    except PolicyError as error:
        print(f"changelog: {error}", file=sys.stderr)
        return 1
    except (OSError, UnicodeError) as error:
        print(f"changelog: operation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
