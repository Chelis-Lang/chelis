"""PEP 517 adapter that seals wheels without sealing editable installs.

It also completes the source distribution. `crates/chelis-std-bundle/build.rs`
packs the chelis-std runtime from `packages/chelis-std`, which is no crate, so
maturin leaves it out of the sdist. The sdist gains every git-tracked file
under `packages/chelis-std`, and the build script selects its inputs from that
copy as it does in a checkout.
"""
from __future__ import annotations

import gzip
import io
from pathlib import Path
import subprocess
import tarfile
from typing import Any

import maturin


ConfigSettings = dict[str, Any] | None


def _wheel_build_args(args: list[str]) -> list[str]:
    # Maturin's CLI feature list replaces [tool.maturin].features.
    features = ["extension-module", "sealed-runtime"]
    forwarded: list[str] = []
    index = 0
    while index < len(args):
        argument = args[index]
        if argument == "--features" and index + 1 < len(args):
            features.extend(args[index + 1].split(","))
            index += 2
            continue
        if argument.startswith("--features="):
            features.extend(argument.removeprefix("--features=").split(","))
        else:
            forwarded.append(argument)
        index += 1
    enabled = dict.fromkeys(feature for feature in features if feature)
    return [*forwarded, "--features", ",".join(enabled)]


def _sealed_wheel_settings(config_settings: ConfigSettings) -> dict[str, Any]:
    settings = dict(config_settings or {})
    args = list(maturin.get_maturin_pep517_args(config_settings) or ())
    settings["maturin.build-args"] = _wheel_build_args(args)
    return settings


def build_wheel(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> str:
    return maturin.build_wheel(
        wheel_directory,
        config_settings=_sealed_wheel_settings(config_settings),
        metadata_directory=metadata_directory,
    )


def prepare_metadata_for_build_wheel(
    metadata_directory: str, config_settings: ConfigSettings = None
) -> str:
    return maturin.prepare_metadata_for_build_wheel(
        metadata_directory,
        config_settings=_sealed_wheel_settings(config_settings),
    )


def build_editable(
    wheel_directory: str,
    config_settings: ConfigSettings = None,
    metadata_directory: str | None = None,
) -> str:
    return maturin.build_editable(
        wheel_directory,
        config_settings=config_settings,
        metadata_directory=metadata_directory,
    )


def prepare_metadata_for_build_editable(
    metadata_directory: str, config_settings: ConfigSettings = None
) -> str:
    return maturin.prepare_metadata_for_build_editable(
        metadata_directory, config_settings=config_settings
    )


RUNTIME_PACKAGE = "packages/chelis-std"


def _workspace_root() -> Path:
    """The checkout this project builds from, two levels above it."""
    here = Path.cwd().resolve()
    for root in (here, *here.parents):
        if (root / RUNTIME_PACKAGE / "reef.toml").is_file():
            return root
    raise RuntimeError(f"no {RUNTIME_PACKAGE}/reef.toml above {here}")


def runtime_package_files(root: Path) -> list[str]:
    """Every git-tracked file under `packages/chelis-std`, root-relative."""
    listing = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", RUNTIME_PACKAGE],
        check=True,
        capture_output=True,
    ).stdout
    files = sorted(path for path in listing.decode("utf-8").split("\0") if path)
    if not files:
        raise RuntimeError(f"git tracks no files under {RUNTIME_PACKAGE} in {root}")
    return files


def add_runtime_package(sdist: Path, root: Path) -> None:
    """Rewrite `sdist` with `runtime_package_files(root)` under its top directory."""
    with tarfile.open(sdist, "r:gz") as existing:
        members = [(member, existing.extractfile(member)) for member in existing.getmembers()]
        members = [
            (member, stream.read() if stream is not None else None)
            for member, stream in members
        ]
    prefix = members[0][0].name.split("/", 1)[0]
    mtime = members[0][0].mtime
    present = {member.name for member, _ in members}
    for relative in runtime_package_files(root):
        name = f"{prefix}/{relative}"
        if name in present:
            continue
        data = (root / relative).read_bytes()
        info = tarfile.TarInfo(name)
        info.size = len(data)
        info.mode = 0o644
        info.mtime = mtime
        members.append((info, data))
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=mtime) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for member, data in members:
                archive.addfile(member, io.BytesIO(data) if data is not None else None)
    sdist.write_bytes(buffer.getvalue())


def build_sdist(sdist_directory: str, config_settings: ConfigSettings = None) -> str:
    name = maturin.build_sdist(sdist_directory, config_settings=config_settings)
    add_runtime_package(Path(sdist_directory) / name, _workspace_root())
    return name


def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> list[str]:
    return maturin.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> list[str]:
    return maturin.get_requires_for_build_editable(config_settings)


def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> list[str]:
    return maturin.get_requires_for_build_sdist(config_settings)
