"""PEP 517 adapter that seals wheels without sealing editable installs.

It also completes the source distribution. `crates/chelis-std-bundle/build.rs`
packs the chelis-std runtime from `packages/chelis-std`, which is no crate, so
maturin leaves it out of the sdist; the sdist gains its `reef.toml` and the
`.ch` files under `src/`, the inputs the build script stages.
"""
from __future__ import annotations

import gzip
import io
import os
from pathlib import Path
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


def _runtime_package() -> Path:
    """`packages/chelis-std` of the workspace this project builds from: two
    levels up in a checkout, at the root of an extracted sdist."""
    here = Path.cwd().resolve()
    for root in (here, *here.parents):
        package = root / "packages" / "chelis-std"
        if (package / "reef.toml").is_file():
            return package
    raise RuntimeError(f"no packages/chelis-std/reef.toml above {here}")


def _runtime_sources(package: Path) -> list[Path]:
    """The manifest and every `.ch` file under `src/`, package-relative."""
    sources = [Path("reef.toml")]
    for directory, _, files in os.walk(package / "src"):
        for name in files:
            if name.endswith(".ch"):
                sources.append((Path(directory) / name).relative_to(package))
    return sorted(sources, key=lambda path: path.as_posix())


def _add_runtime_sources(sdist: Path) -> None:
    """Rewrite `sdist` with the runtime sources under its workspace root."""
    with tarfile.open(sdist, "r:gz") as existing:
        members = [(member, existing.extractfile(member)) for member in existing.getmembers()]
        members = [
            (member, stream.read() if stream is not None else None)
            for member, stream in members
        ]
    prefix = members[0][0].name.split("/", 1)[0]
    mtime = members[0][0].mtime
    present = {member.name for member, _ in members}
    package = _runtime_package()
    for relative in _runtime_sources(package):
        name = f"{prefix}/packages/chelis-std/{relative.as_posix()}"
        if name in present:
            continue
        data = (package / relative).read_bytes()
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
    _add_runtime_sources(Path(sdist_directory) / name)
    return name


def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> list[str]:
    return maturin.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> list[str]:
    return maturin.get_requires_for_build_editable(config_settings)


def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> list[str]:
    return maturin.get_requires_for_build_sdist(config_settings)
