"""Independent expected values for the shared compiled-library JSON manifest."""

from capacity_census_wire_adapters import CodecCase, canonical


def artifact_cases():
    cases = []

    def add(carrier, codec, name, text, expected, error=None):
        cases.append(
            CodecCase(
                f"{carrier}/{codec}/{name}",
                "artifact",
                carrier,
                codec,
                text,
                expected,
                error,
            )
        )

    for version in (0, 1, 2, 3, 4294967295):
        for codec in ("json", "construct"):
            add(
                "ArtifactAbiVersion",
                codec,
                str(version),
                str(version),
                2 if version == 2 else None,
                None if version == 2 else "artifact ABI version",
            )
    for index, text in enumerate(("null", "true", '"1"', "-1", "1.0", "4294967296")):
        add("ArtifactAbiVersion", "json", f"invalid-{index}", text, None)

    base = {
        "abi_version": 2,
        "target": "c",
        "host_entry_name": "chelis_main",
        "inputs": [],
        "outputs": [],
        "source_path": "model.chelis",
        "source_hash": "digest",
    }
    for size in (0, 1, 9007199254740993, 9223372036854775807):
        expected = {
            **base,
            "inputs": [
                {"name": "x", "dtype": "float32", "dims": [{"name": "n", "size": size}]}
            ],
        }
        add(
            "CompiledArtifactManifest",
            "json",
            f"extent-{size}",
            canonical(expected),
            expected,
        )
        add(
            "CompiledArtifactManifest",
            "construct",
            f"extent-{size}",
            canonical({"version": 2, "size": size}),
            expected,
        )
    add("CompiledArtifactManifest", "json", "empty", canonical(base), base)
    optional = {**base, "device_entry_name": "device", "symbolic_dims": ["n"]}
    add("CompiledArtifactManifest", "json", "optional", canonical(optional), optional)
    for version in (0, 1, 3, 4294967295):
        add(
            "CompiledArtifactManifest",
            "construct",
            f"version-{version}",
            canonical({"version": version, "size": 1}),
            None,
            "artifact ABI version",
        )
    for size in (-1, -9223372036854775808):
        add(
            "CompiledArtifactManifest",
            "construct",
            f"extent-{size}",
            canonical({"version": 2, "size": size}),
            None,
            "nonnegative",
        )
    for index, header in enumerate(
        (
            "",
            ',"abi_version":0',
            ',"abi_version":1',
            ',"abi_version":3',
            ',"abi_version":-1',
            ',"abi_version":2.0',
            ',"abi_version":true',
            ',"abi_version":"2"',
            ',"abi_version":4294967295',
            ',"abi_version":2,"abi_version":2',
        )
    ):
        add(
            "CompiledArtifactManifest",
            "json",
            f"version-before-metadata-{index}",
            '{"inputs":"invalid"' + header + "}",
            None,
            "artifact ABI version",
        )
    for index, size in enumerate((-1, 9223372036854775808, 1.0, True)):
        bad = {
            **base,
            "inputs": [
                {"name": "x", "dtype": "float32", "dims": [{"name": "n", "size": size}]}
            ],
        }
        add(
            "CompiledArtifactManifest",
            "json",
            f"invalid-extent-{index}",
            canonical(bad),
            None,
        )
    return cases
