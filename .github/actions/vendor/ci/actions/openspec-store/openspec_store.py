#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

MAX_LOCK_BYTES = 16 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_SPEC_BYTES = 1024 * 1024
MAX_TRACKED_FILE_BYTES = 2 * 1024 * 1024
LOCK_FIELDS = frozenset({"version", "id", "repository", "commit"})
LOCK_LINE = re.compile(r"^([a-z][a-z0-9_-]*): ([^\x00-\x20][\x20-\x7e]*)$")
LOCK_INTEGER = re.compile(r"^(0|[1-9][0-9]*)$")
COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_OPENSPEC_VERSION = "1.6.0"
TOKEN_ENVIRONMENT = "CHELIS_OPENSPEC_STORE_TOKEN"
LOCAL_SPEC_MARKER = "openspec" + "/specs/"
DEPENDENCY_SOURCE_SUFFIXES = frozenset({".nix", ".py", ".sh", ".yaml", ".yml"})
SETUP_CONTRACT_MARKER = "This change SHALL NOT claim or perform a consumer migration"


class StoreError(ValueError):
    pass


class StoreLockVersion(IntEnum):
    V1 = 1


class StoreId(StrEnum):
    CHELIS_PLANS = "chelis-plans"


class StoreRepository(StrEnum):
    CHELIS_OPENSPEC = "Chelis-Lang/openspec"


class RootSource(StrEnum):
    DECLARED = "declared"


class SpecificationId(StrEnum):
    SHARED_CHELIS_SETUP = "shared-chelis-setup"


class CommitSha(str):
    def __new__(cls, raw: str) -> CommitSha:
        if COMMIT_SHA.fullmatch(raw) is None:
            raise StoreError("The store commit must be one 40-character lowercase SHA.")
        return cast(CommitSha, str.__new__(cls, raw))


class OpenSpecVersion(str):
    def __new__(cls, raw: str) -> OpenSpecVersion:
        if raw != EXPECTED_OPENSPEC_VERSION:
            raise StoreError(
                f"The OpenSpec version must be {EXPECTED_OPENSPEC_VERSION}."
            )
        return cast(OpenSpecVersion, str.__new__(cls, raw))


@dataclass(frozen=True, slots=True, repr=False)
class StoreToken:
    value: str

    @classmethod
    def parse(cls, raw: str) -> StoreToken:
        if (
            not raw
            or len(raw) > 1024
            or raw != raw.strip()
            or not raw.isascii()
            or not raw.isprintable()
        ):
            raise StoreError("The store access token is absent or malformed.")
        return cls(raw)


@dataclass(frozen=True, slots=True)
class StoreLockPath:
    value: Path

    @classmethod
    def parse(cls, raw: str) -> StoreLockPath:
        path = Path(raw)
        if (
            not raw
            or len(raw) > 1024
            or raw != raw.strip()
            or "\\" in raw
            or path.is_absolute()
            or path.as_posix() != raw
            or any(part in ("", ".", "..") for part in path.parts)
        ):
            raise argparse.ArgumentTypeError(
                "The store lock path must be a normalized repository-relative path."
            )
        return cls(path)


@dataclass(frozen=True, slots=True)
class StoreRevision:
    version: StoreLockVersion
    store_id: StoreId
    repository: StoreRepository
    commit: CommitSha

    def to_json_object(self) -> dict[str, object]:
        return {
            "version": int(self.version),
            "id": self.store_id.value,
            "repository": self.repository.value,
            "commit": str(self.commit),
        }


@dataclass(frozen=True, slots=True)
class VerifiedStoreCheckout:
    revision: StoreRevision
    path: Path
    head: CommitSha


@dataclass(frozen=True, slots=True)
class StoreHealth:
    root_path: Path
    root_source: RootSource
    store_id: StoreId


@dataclass(frozen=True, slots=True)
class StoreAccess:
    repository: StoreRepository
    commit: CommitSha
    read_only: bool


@dataclass(frozen=True, slots=True)
class ApiResponse:
    status: int
    body: bytes


ApiRequest = Callable[[str, StoreToken], ApiResponse]


def _parse_lock_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = LOCK_LINE.fullmatch(line)
        if match is None:
            raise StoreError("The store lock must use one plain scalar per line.")
        name, value = match.groups()
        if name in fields:
            raise StoreError(f"The store lock field is duplicated: {name}")
        fields[name] = value
    if set(fields) != LOCK_FIELDS:
        raise StoreError(
            "The store lock fields must be exactly: commit, id, repository, version."
        )
    return fields


def parse_store_lock(text: str) -> StoreRevision:
    mapping = _parse_lock_fields(text)
    raw_version = mapping["version"]
    if LOCK_INTEGER.fullmatch(raw_version) is None:
        raise StoreError("The store lock field must be an integer: version")
    try:
        version = StoreLockVersion(int(raw_version))
    except ValueError as error:
        raise StoreError("The store lock version must be 1.") from error

    try:
        store_id = StoreId(mapping["id"])
    except ValueError as error:
        raise StoreError("The store ID must be chelis-plans.") from error

    try:
        repository = StoreRepository(mapping["repository"])
    except ValueError as error:
        raise StoreError(
            "The store repository must be Chelis-Lang/openspec."
        ) from error

    return StoreRevision(
        version=version,
        store_id=store_id,
        repository=repository,
        commit=CommitSha(mapping["commit"]),
    )


def _read_text_file(path: Path, maximum: int, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise StoreError(f"The {label} must be a regular non-symlink file.")
    try:
        size = path.stat().st_size
    except OSError as error:
        raise StoreError(f"The {label} cannot be inspected.") from error
    if size > maximum:
        raise StoreError(f"The {label} exceeds the size limit.")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise StoreError(f"The {label} must contain UTF-8 text.") from error


def load_store_lock(path: Path) -> StoreRevision:
    return parse_store_lock(_read_text_file(path, MAX_LOCK_BYTES, "store lock"))


def parse_checkout_head(raw: str) -> CommitSha:
    if not raw.endswith("\n") or raw.count("\n") != 1:
        raise StoreError("The store checkout HEAD output is not canonical.")
    return CommitSha(raw[:-1])


def read_checkout_head(checkout: Path) -> CommitSha:
    if checkout.is_symlink() or not checkout.is_dir():
        raise StoreError("The store checkout must be a regular non-symlink directory.")
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise StoreError("The store checkout HEAD cannot be read.") from error
    if completed.returncode != 0:
        raise StoreError("The store checkout HEAD cannot be read.")
    return parse_checkout_head(completed.stdout)


def verify_store_checkout(
    revision: StoreRevision, checkout: Path
) -> VerifiedStoreCheckout:
    actual = read_checkout_head(checkout)
    if actual != revision.commit:
        raise StoreError(
            "The store checkout HEAD does not match the locked commit: "
            f"expected {revision.commit}, found {actual}"
        )
    try:
        canonical = checkout.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise StoreError("The store checkout path cannot be resolved.") from error
    return VerifiedStoreCheckout(revision=revision, path=canonical, head=actual)


def require_store_token(environment: Mapping[str, str]) -> StoreToken:
    return StoreToken.parse(environment.get(TOKEN_ENVIRONMENT, ""))


def _bounded_body(stream: object) -> bytes:
    reader = getattr(stream, "read", None)
    if not callable(reader):
        raise StoreError("The GitHub API response cannot be read.")
    value = cast(bytes, reader(MAX_JSON_BYTES + 1))
    if len(value) > MAX_JSON_BYTES:
        raise StoreError("The GitHub API response exceeds the size limit.")
    return value


def github_request(url: str, token: StoreToken) -> ApiResponse:
    request = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token.value}",
            "User-Agent": "Chelis-Lang-openspec-store-action",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=15) as response:
            return ApiResponse(status=response.status, body=_bounded_body(response))
    except HTTPError as error:
        try:
            return ApiResponse(status=error.code, body=_bounded_body(error))
        finally:
            error.close()
    except (OSError, URLError) as error:
        raise StoreError("The GitHub API request failed.") from error


def _api_object(response: ApiResponse, label: str) -> dict[str, object]:
    if response.status != 200:
        raise StoreError(
            f"The GitHub API cannot access the {label}: HTTP {response.status}"
        )
    try:
        value = cast(object, json.loads(response.body.decode("utf-8")))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise StoreError(f"The GitHub API returned invalid {label} JSON.") from error
    if not isinstance(value, dict):
        raise StoreError(f"The GitHub API returned invalid {label} JSON.")
    return cast(dict[str, object], value)


def require_store_access(
    revision: StoreRevision,
    token: StoreToken,
    request: ApiRequest = github_request,
) -> StoreAccess:
    listing = _api_object(
        request("https://api.github.com/installation/repositories", token),
        "store installation",
    )
    if listing.get("repository_selection") != "selected":
        raise StoreError("The store token is not limited to selected repositories.")
    repositories = listing.get("repositories")
    if not isinstance(repositories, list) or len(repositories) != 1:
        raise StoreError("The store token is not limited to one repository.")
    only = repositories[0]
    if not isinstance(only, dict) or only.get("full_name") != revision.repository.value:
        raise StoreError("The store token is not limited to the store repository.")

    base = f"https://api.github.com/repos/{revision.repository.value}"
    commit = _api_object(
        request(f"{base}/git/commits/{revision.commit}", token),
        "locked store commit",
    )
    actual = commit.get("sha")
    if type(actual) is not str or CommitSha(actual) != revision.commit:
        raise StoreError("The GitHub API returned the wrong store commit.")

    return StoreAccess(
        repository=revision.repository,
        commit=revision.commit,
        read_only=True,
    )


def _json_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise StoreError(f"The OpenSpec doctor field must be an object: {name}")
    return cast(dict[str, object], value)


def _require_no_error_status(mapping: Mapping[str, object], name: str) -> None:
    status = mapping.get("status")
    if not isinstance(status, list):
        raise StoreError(f"The OpenSpec doctor status must be an array: {name}")
    for raw_entry in status:
        entry = _json_mapping(raw_entry, f"{name}.status")
        severity = entry.get("severity")
        if severity not in ("info", "warning", "error"):
            raise StoreError(f"The OpenSpec doctor status severity is invalid: {name}")
        if severity == "error":
            raise StoreError(f"The OpenSpec doctor reports an error: {name}")


def parse_store_health(text: str) -> StoreHealth:
    try:
        value = cast(object, json.loads(text))
    except json.JSONDecodeError as error:
        raise StoreError("The OpenSpec doctor output must be valid JSON.") from error
    document = _json_mapping(value, "document")
    root = _json_mapping(document.get("root"), "root")
    store = _json_mapping(document.get("store"), "store")
    _require_no_error_status(document, "document")
    _require_no_error_status(root, "root")
    _require_no_error_status(store, "store")

    raw_path = root.get("path")
    if type(raw_path) is not str or not Path(raw_path).is_absolute():
        raise StoreError("The OpenSpec doctor root path must be absolute.")
    try:
        source = RootSource(root.get("source"))
        root_id = StoreId(root.get("store_id"))
        store_id = StoreId(store.get("id"))
    except (TypeError, ValueError) as error:
        raise StoreError("The OpenSpec doctor store identity is invalid.") from error
    if root.get("healthy") is not True or root_id is not store_id:
        raise StoreError("The OpenSpec doctor root is not healthy.")
    return StoreHealth(
        root_path=Path(raw_path),
        root_source=source,
        store_id=root_id,
    )


def load_store_health(path: Path) -> StoreHealth:
    return parse_store_health(
        _read_text_file(path, MAX_JSON_BYTES, "OpenSpec doctor output")
    )


def require_store_health(checkout: VerifiedStoreCheckout, health: StoreHealth) -> None:
    if health.root_path != checkout.path:
        raise StoreError("The OpenSpec doctor root does not match the store checkout.")
    if health.store_id is not checkout.revision.store_id:
        raise StoreError("The OpenSpec doctor store ID does not match the store lock.")


def specification_path(
    checkout: VerifiedStoreCheckout, specification: SpecificationId
) -> Path:
    return checkout.path / "openspec" / "specs" / specification.value / "spec.md"


def require_setup_contract(checkout: VerifiedStoreCheckout) -> Path:
    path = specification_path(checkout, SpecificationId.SHARED_CHELIS_SETUP)
    text = _read_text_file(path, MAX_SPEC_BYTES, "shared-chelis-setup specification")
    if SETUP_CONTRACT_MARKER not in text:
        raise StoreError(
            "The shared-chelis-setup specification omits the migration claim rule."
        )
    return path


def local_spec_dependencies(files: Mapping[Path, str]) -> tuple[Path, ...]:
    return tuple(
        sorted(path for path, text in files.items() if LOCAL_SPEC_MARKER in text)
    )


def tracked_local_spec_dependencies(root: Path) -> tuple[Path, ...]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=False,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise StoreError("The tracked repository files cannot be listed.") from error
    if completed.returncode != 0:
        raise StoreError("The tracked repository files cannot be listed.")

    files: dict[Path, str] = {}
    for raw_path in completed.stdout.split(b"\0"):
        if not raw_path:
            continue
        try:
            relative = Path(raw_path.decode("utf-8"))
            if relative.suffix not in DEPENDENCY_SOURCE_SUFFIXES:
                continue
            candidate = root / relative
            if candidate.is_symlink() or not candidate.is_file():
                continue
            if candidate.stat().st_size > MAX_TRACKED_FILE_BYTES:
                continue
            value = candidate.read_bytes()
        except (OSError, UnicodeError) as error:
            raise StoreError("A tracked repository file cannot be read.") from error
        try:
            files[relative] = value.decode("utf-8")
        except UnicodeError:
            continue
    return local_spec_dependencies(files)


def require_no_local_spec_dependencies(root: Path) -> None:
    dependencies = tracked_local_spec_dependencies(root)
    if dependencies:
        names = ", ".join(str(path) for path in dependencies)
        raise StoreError(
            f"Tracked files use local OpenSpec specification paths: {names}"
        )


def _append_file(path: Path, text: str, label: str) -> None:
    try:
        with path.open("a", encoding="utf-8") as output:
            output.write(text)
    except OSError as error:
        raise StoreError(f"The {label} cannot be written.") from error


def append_github_output(
    path: Path,
    revision: StoreRevision,
    checkout: VerifiedStoreCheckout | None = None,
    health: StoreHealth | None = None,
    version: OpenSpecVersion | None = None,
) -> None:
    text = (
        f"store-id={revision.store_id.value}\n"
        f"repository={revision.repository.value}\n"
        f"commit={revision.commit}\n"
    )
    evidence = (checkout, health, version)
    if any(value is not None for value in evidence):
        if checkout is None or health is None or version is None:
            raise StoreError("The GitHub outputs require complete store evidence.")
        text += (
            f"root-path={checkout.path}\n"
            f"root-source={health.root_source.value}\n"
            f"openspec-version={version}\n"
        )
    _append_file(path, text, "GitHub output file")


def append_github_summary(
    path: Path,
    version: OpenSpecVersion,
    checkout: VerifiedStoreCheckout,
    health: StoreHealth,
) -> None:
    _append_file(
        path,
        "## OpenSpec shared store\n\n"
        f"- OpenSpec version: `{version}`\n"
        f"- Store ID: `{health.store_id.value}`\n"
        f"- Root source: `{health.root_source.value}`\n"
        f"- Store commit: `{checkout.head}`\n",
        "GitHub summary file",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the immutable OpenSpec store contract"
    )
    parser.add_argument("--lock", type=StoreLockPath.parse, required=True)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--doctor-json", type=Path)
    parser.add_argument("--openspec-version")
    parser.add_argument("--github-output", type=Path)
    parser.add_argument("--github-summary", type=Path)
    parser.add_argument("--check-access", action="store_true")
    parser.add_argument("--assert-setup-contract", action="store_true")
    parser.add_argument("--check-local-spec-dependencies", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        revision = load_store_lock(arguments.lock.value)
        checkout = None
        if arguments.check_access:
            require_store_access(revision, require_store_token(os.environ))
        if arguments.checkout is not None:
            checkout = verify_store_checkout(revision, arguments.checkout)
        if arguments.assert_setup_contract:
            if checkout is None:
                raise StoreError("The setup contract check requires a checkout.")
            require_setup_contract(checkout)
        if arguments.check_local_spec_dependencies:
            require_no_local_spec_dependencies(Path.cwd())

        health = None
        if arguments.doctor_json is not None:
            if checkout is None:
                raise StoreError("The doctor check requires a checkout.")
            health = load_store_health(arguments.doctor_json)
            require_store_health(checkout, health)

        version = None
        if arguments.openspec_version is not None:
            version = OpenSpecVersion(arguments.openspec_version)
        if arguments.github_summary is not None:
            if checkout is None or health is None or version is None:
                raise StoreError(
                    "The GitHub summary requires a checkout, doctor output, and version."
                )
            append_github_summary(arguments.github_summary, version, checkout, health)
        if arguments.github_output is not None:
            append_github_output(
                arguments.github_output,
                revision,
                checkout,
                health,
                version,
            )

        print(json.dumps(revision.to_json_object(), sort_keys=True))
        return 0
    except StoreError as error:
        print(f"openspec-store: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
