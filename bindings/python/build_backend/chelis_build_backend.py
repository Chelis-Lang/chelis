"""PEP 517 adapter that seals wheels without sealing editable installs."""
from __future__ import annotations

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


def build_sdist(sdist_directory: str, config_settings: ConfigSettings = None) -> str:
    return maturin.build_sdist(sdist_directory, config_settings=config_settings)


def get_requires_for_build_wheel(config_settings: ConfigSettings = None) -> list[str]:
    return maturin.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(config_settings: ConfigSettings = None) -> list[str]:
    return maturin.get_requires_for_build_editable(config_settings)


def get_requires_for_build_sdist(config_settings: ConfigSettings = None) -> list[str]:
    return maturin.get_requires_for_build_sdist(config_settings)
