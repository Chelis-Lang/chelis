"""Close rustdoc's block-local declaration hole with compiler expansion evidence.

C6's published graph cannot discover a function/constant-local nominal type or
impl. The structural rule here requires those declarations at module scope,
including non-serialized ones. The only local implementation machinery admitted
is emitted by the compiler-resolved serde_derive::Serialize/Deserialize macros,
or the locked schemars_derive::JsonSchema's impls and unit marker structs.
The complete AST is still visited inside that machinery; an enclosing generated
item never confers its provenance on a user-authored child.

This is not publication-call or dataflow analysis. In particular, direct
serialization of primitives, containers, or serde_json::Value has no new nominal
declaration and requires separate publication ownership. Evidence covers exactly
the library configuration passed to expand_library, not inactive cfg branches.
The pinned compiler's verbose hygiene format is required; missing/changed
provenance is an error, never a reason to accept a declaration.
"""

from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


@dataclass(frozen=True)
class ExpandedLibrary:
    source: str
    serde_crate: str
    command: tuple[str, ...]
    serde_package: str
    schema_crate: str | None = None
    schema_package: str | None = None


@dataclass(frozen=True)
class LocalClosure:
    expanded_identity: str
    module_nominals: int
    generated_local_items: int
    serde_derives: int
    schema_derives: int


def _one(pattern: str, text: str, description: str) -> re.Match:
    matches = list(re.finditer(pattern, text, re.MULTILINE | re.DOTALL))
    if len(matches) != 1:
        raise ValueError(f"missing or ambiguous compiler {description}")
    return matches[0]


def compiler_provenance(
    expanded: str, serde_crate: str, schema_crate: str | None = None
) -> tuple[str, dict[str, str], int, int]:
    """Read rustc metadata, not source spellings of derive names or attributes."""
    marker = "\n/*\nHygieneData {\n"
    if marker not in expanded or not expanded.endswith("}\n*/\n"):
        raise ValueError("missing compiler verbose hygiene artifact")
    source, metadata = expanded.rsplit(marker, 1)
    local = _one(
        r"^    local_expn_data: \[\n(.*?)^    \],\n", metadata, "expansion table"
    )[1]
    entries = list(
        re.finditer(
            r"        Some\(\n            ExpnData \{\n(.*?)"
            r"            \},\n        \),\n",
            local,
            re.DOTALL,
        )
    )
    if not entries or "".join(match[0] for match in entries) != local:
        raise ValueError("unsupported or incomplete compiler expansion records")
    parents = []
    derives = {}
    serde_derives = 0
    schema_derives = 0
    for index, match in enumerate(entries):
        record = match[1]
        parent = _one(
            r"^                parent: crate(\d+)::\{\{expn(\d+)\}\},$",
            record,
            "expansion parent",
        )
        parents.append((int(parent[1]), int(parent[2])))
        definition = _one(
            r"^                macro_def_id: (None,|Some\(\n"
            r"                    DefId\(([^\n]+)\),\n                \),)$",
            record,
            "macro definition",
        )[2]
        kind = _one(
            r"^                kind: (.*?)(?=^                parent:)",
            record,
            "expansion kind",
        )[1]
        if definition is not None:
            canonical = re.fullmatch(
                r"([1-9]\d*):\d+ ~ "
                + re.escape(serde_crate)
                + r"::(Serialize|Deserialize)",
                definition,
            )
            schema = schema_crate is not None and re.fullmatch(
                r"([1-9]\d*):\d+ ~ " + re.escape(schema_crate) + r"::JsonSchema",
                definition,
            )
            if (canonical or schema) and re.fullmatch(
                r'Macro\(\n                    Derive,\n                    "[^"\n]+",\n                \),\n',
                kind,
            ):
                derives[index] = definition
                serde_derives += int(canonical is not None)
                schema_derives += int(bool(schema))
    if parents[0] != (0, 0) or "kind: Root," not in entries[0][1]:
        raise ValueError("invalid compiler expansion root")
    # Every admitted expansion must have a complete, acyclic local ancestry.
    for index in derives:
        seen = set()
        current = index
        while current:
            if current in seen or current >= len(parents):
                raise ValueError("broken compiler expansion ancestry")
            seen.add(current)
            crate, current = parents[current]
            if crate != 0:
                raise ValueError("unsupported foreign parent of local serde expansion")
    contexts = _one(
        r"^    syntax_context_data: \[\n(.*?)^    \],\n", metadata, "syntax contexts"
    )[1]
    rows = list(
        re.finditer(
            r"        SyntaxContextData \{\n"
            r"            outer_expn: crate(\d+)::\{\{expn(\d+)\}\},\n"
            r"            outer_transparency: (?:Opaque|SemiOpaque|Transparent),\n"
            r"            parent: #(\d+),\n"
            r"            opaque: #(\d+),\n"
            r"            opaque_and_semiopaque: #(\d+),\n"
            r'            dollar_crate_name: "[^"\n]*",\n'
            r"        \},\n",
            contexts,
        )
    )
    if not rows or "".join(row[0] for row in rows) != contexts:
        raise ValueError("unsupported or incomplete compiler syntax contexts")
    recognized = {}
    for index, row in enumerate(rows):
        if any(int(row[n]) >= len(rows) for n in (3, 4, 5)):
            raise ValueError("unresolved compiler syntax context")
        if index and int(row[3]) >= index:
            raise ValueError("cyclic compiler syntax context")
        crate, expansion = int(row[1]), int(row[2])
        if crate == 0 and expansion >= len(entries):
            raise ValueError("unresolved compiler syntax expansion")
        if crate == 0 and expansion in derives:
            recognized[str(index)] = derives[expansion]
    return source, recognized, serde_derives, schema_derives


def _run(command: list[str], *, root: Path, target: Path, stdin=None) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        input=stdin,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RUSTC_BOOTSTRAP": "1",
            "CARGO_BUILD_JOBS": "1",
            "CARGO_HUSKY_DONT_INSTALL_HOOKS": "1",
            "CARGO_TARGET_DIR": str(target),
            "PYO3_PYTHON": sys.executable,
            "VIRTUAL_ENV": sys.prefix,
        },
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"local publication command failed ({result.returncode}): "
            f"{' '.join(command)}\n{result.stderr}"
        )
    return result.stdout


def build_probe(root: Path, target: Path) -> Path:
    _run(
        [
            "cargo",
            "build",
            "--locked",
            "-p",
            "chelis-compiler-api",
            "--example",
            "wire_publication_probe",
        ],
        root=root,
        target=target,
    )
    return target / "debug/examples/wire_publication_probe"


def _macro_crate(artifacts: list[dict], name: str, *, root: Path, target: Path):
    selected = [
        artifact for artifact in artifacts if artifact["target"]["name"] == name
    ]
    if len(selected) != 1:
        raise ValueError(f"missing or ambiguous {name} dependency artifact")
    artifact = selected[0]
    package = artifact["package_id"]
    if (
        not re.fullmatch(
            r"registry\+https://github.com/rust-lang/crates.io-index#"
            + name
            + r"@[0-9.]+",
            package,
        )
        or artifact["target"]["kind"] != ["proc-macro"]
        or len(artifact["filenames"]) != 1
    ):
        raise ValueError(f"unsupported {name} dependency origin")
    crate_info = _run(
        ["rustc", "-Zls=root", artifact["filenames"][0]], root=root, target=target
    )
    stable_id = int(
        _one(
            r"^hash [0-9a-f]+ stable_crate_id StableCrateId\((\d+)\)$",
            crate_info,
            f"{name} crate identity",
        )[1]
    )
    # DefId's pretty form uses the first four hex digits of StableCrateId.
    # Cargo origin/uniqueness above prevents another same-named crate from
    # receiving admission through this deliberately shortened spelling.
    prefix = f"{stable_id:016x}"[:4]
    return f"{name}[{prefix}]", package


def expand_library(
    root: Path,
    target: Path,
    *,
    package: str | None = None,
    manifest: Path | None = None,
    rustc_args: tuple[str, ...] = (),
) -> ExpandedLibrary:
    """Expand exactly one real library, including macros and active cfg branches."""
    command = ["cargo", "rustc", "--locked", "--lib", "--message-format=json"]
    if package is not None:
        command += ["-p", package]
    if manifest is not None:
        command += ["--manifest-path", str(manifest)]
    with tempfile.TemporaryDirectory(
        prefix="wire-expansion-", dir=root / "target"
    ) as tmp:
        output = Path(tmp) / "expanded.rs"
        command += [
            "--",
            "-Zunpretty=expanded,hygiene",
            "-Zverbose-internals",
            "-o",
            str(output),
            *rustc_args,
        ]
        messages = _run(command, root=root, target=target)
        artifacts = []
        try:
            for line in messages.splitlines():
                message = json.loads(line)
                if message.get("reason") == "compiler-artifact" and message["target"][
                    "name"
                ] in ("serde_derive", "schemars_derive"):
                    artifacts.append(message)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError("invalid compiler artifact stream") from error
        serde_crate, package = _macro_crate(
            artifacts, "serde_derive", root=root, target=target
        )
        schema_crate, schema_package = None, None
        if any(a["target"]["name"] == "schemars_derive" for a in artifacts):
            schema_crate, schema_package = _macro_crate(
                artifacts, "schemars_derive", root=root, target=target
            )
        if not output.is_file():
            raise ValueError("missing compiler expanded library")
        return ExpandedLibrary(
            output.read_text(),
            serde_crate,
            tuple(command),
            package,
            schema_crate,
            schema_package,
        )


def verify_expanded_library(
    expanded: ExpandedLibrary,
    probe: Path,
    *,
    root: Path,
    target: Path,
) -> LocalClosure:
    if not isinstance(expanded, ExpandedLibrary):
        raise ValueError("missing compiler expansion/dependency evidence")
    source, contexts, derives, schema_derives = compiler_provenance(
        expanded.source, expanded.serde_crate, expanded.schema_crate
    )
    response = _run(
        [str(probe)],
        root=root,
        target=target,
        stdin=json.dumps({"source": source, "derive_contexts": contexts}),
    )
    try:
        observation = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("invalid local publication probe output") from error
    if set(observation) != {"module_nominals", "generated_local_items", "findings"}:
        raise ValueError("incomplete local publication probe output")
    if not isinstance(observation["findings"], list):
        raise ValueError("invalid local publication findings")
    if observation["findings"]:
        raise ValueError("; ".join(observation["findings"]))
    for name in ("module_nominals", "generated_local_items"):
        if type(observation[name]) is not int or observation[name] < 0:
            raise ValueError("invalid local publication counts")
    return LocalClosure(
        hashlib.sha256(expanded.source.encode()).hexdigest(),
        observation["module_nominals"],
        observation["generated_local_items"],
        derives,
        schema_derives,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    target = args.target or root / "target/agents/wire-codec-rustdoc"
    target = target.resolve()
    try:
        probe = build_probe(root, target)
        expanded = expand_library(root, target, package="chelis-compiler-api")
        result = verify_expanded_library(expanded, probe, root=root, target=target)
    except (OSError, ValueError) as error:
        print(f"WIRE LOCAL PUBLICATION: FAIL: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                **asdict(result),
                "command": expanded.command,
                "serde_package": expanded.serde_package,
                "schema_package": expanded.schema_package,
            },
            sort_keys=True,
        )
    )
    print("WIRE LOCAL PUBLICATION: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
