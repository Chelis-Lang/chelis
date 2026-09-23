#!/usr/bin/env python3
"""Drive real stable Cargo with exact runtime-closure observation.

The helper is bootstrapped outside producing build steps in a distinct target.
Cargo stdout stays machine-readable for JSON callers; diagnostics and status are
not replaced by observer chatter. No unstable unit graph or archive search.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import runtime_identity_observer as observer


def install_cargo_launcher(directory, *, real_cargo, python):
    """Install a controlled PATH launcher and return its executable Path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "cargo"
    real = str(Path(real_cargo).absolute())
    if Path(real).resolve() == path.resolve():
        raise observer.ObservationError("Cargo launcher cannot target itself")
    script = Path(__file__).resolve()
    text = f"#!{python}\n# chelis-runtime-identity-cargo-launcher\nimport os, sys\nos.execv({str(python)!r}, [{str(python)!r}, {str(script)!r}, '--cargo', {real!r}, '--', *sys.argv[1:]])\n"
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def cargo_command(arguments):
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument in {"--color", "--config", "-C", "-Z"}:
            if index + 1 == len(arguments):
                raise observer.ObservationError(f"missing value for Cargo global option {argument}")
            index += 2
        elif argument.startswith(("-", "+")):
            index += 1
        else:
            return argument
    return ""


def message_format(arguments):
    for index, arg in enumerate(arguments):
        if arg == "--message-format":
            return arguments[index + 1]
        if arg.startswith("--message-format="):
            return arg.split("=", 1)[1]
    return None


def metadata_arguments(arguments):
    result = []
    index = 0
    while index < len(arguments):
        arg = arguments[index]
        if arg in {"--manifest-path", "--config"}:
            result.extend(arguments[index:index + 2]); index += 2
        elif arg.startswith(("--manifest-path=", "--config=")) or arg in {"--locked", "--offline", "--frozen"}:
            result.append(arg); index += 1
        else:
            index += 1
    return result


def event_receipt(event, state):
    found = None
    for filename in event["filenames"]:
        actual = observer.digest(Path(filename).read_bytes())
        # Cargo uplifts both primary libraries and build-script executables.
        # Join the exact reported bytes to indexed compiler observations; never
        # infer their origin from a directory scan, a salt, or modification time.
        paths = set()
        binding = state / "bindings" / (observer.key(filename) + ".json")
        if binding.exists():
            paths.add(observer.load(binding)["observation"])
        index = state / "output-digests" / actual
        for binding in index.glob("*.json"):
            paths.add(observer.load(binding)["observation"])
        matches = set()
        for path in paths:
            receipt = observer.load(path)
            if not any(output["digest"] == actual for output in receipt["outputs"]):
                continue
            if event.get("manifest_path") and receipt.get("manifest_path") != event["manifest_path"]:
                continue
            matches.add(path)
        if not matches:
            return None
        found = matches if found is None else found & matches
        if not found:
            raise observer.ObservationError("one Cargo artifact event names different compilation receipts")
    if found is None:
        return None
    if len(found) != 1:
        raise observer.ObservationError("Cargo artifact bytes have ambiguous compilation observations")
    return next(iter(found))


def run(cargo, arguments):
    environment = dict(os.environ)
    # Cargo help, metadata, fmt, clean, version and external workspaces retain the
    # original behavior; declaring a managed producer still requires this driver.
    command = cargo_command(arguments)
    if command not in {"build", "b", "check", "c", "test", "t", "run", "r", "rustc", "bench"}:
        external = shutil.which("cargo-" + command) if command else None
        if external:
            # Cargo overwrites CARGO with its own executable when dispatching
            # plugins. Execute the plugin directly so its nested builds enter
            # the managed driver rather than bypass observation.
            if arguments[0] != command:
                raise observer.ObservationError("place Cargo plugin options after the plugin name")
            with tempfile.TemporaryDirectory(prefix="chelis-cargo-plugin-") as directory:
                launcher = install_cargo_launcher(directory, real_cargo=cargo, python=sys.executable)
                environment.update({"CARGO": str(launcher), "CHELIS_IDENTITY_REAL_CARGO": cargo,
                                    "PATH": str(launcher.parent) + os.pathsep + environment.get("PATH", "")})
                return subprocess.call([external, *arguments], env=environment, close_fds=False)
        return subprocess.call([cargo, *arguments], env=environment, close_fds=False)
    if sys.version_info < (3, 11):
        raise observer.ObservationError("managed runtime identity builds require Python 3.11 or newer")
    metadata_proc = subprocess.run([cargo, "metadata", "--format-version=1", "--all-features", *metadata_arguments(arguments)], env=environment, text=True, stdout=subprocess.PIPE, check=False)
    if metadata_proc.returncode:
        return metadata_proc.returncode
    metadata = json.loads(metadata_proc.stdout)
    workspace = Path(metadata["workspace_root"])
    if not any(package["name"] == "chelis-runtime" for package in metadata["packages"]):
        return subprocess.call([cargo, *arguments], env=environment, close_fds=False)
    explicit_root = environment.get("CHELIS_IDENTITY_WORKSPACE")
    inherited_root = environment.get("CHELIS_IDENTITY_OBSERVED_WORKSPACE")
    if explicit_root and explicit_root != inherited_root and (not Path(explicit_root).is_dir() or Path(explicit_root).resolve() != workspace.resolve()):
        raise observer.ObservationError("explicit CHELIS_IDENTITY_WORKSPACE is missing or differs from Cargo's actual workspace root")
    source = Path(__file__).resolve().parents[1]
    state = Path(environment.get("CHELIS_IDENTITY_STATE", str(Path(metadata["target_directory"]) / "runtime-identity-observations"))).absolute()
    state.mkdir(parents=True, exist_ok=True)
    helper_target = state / "helper-target"
    bootstrap_env = {key: value for key, value in environment.items() if not key.startswith("CHELIS_IDENTITY_")}
    bootstrap_env.pop("RUSTC_WRAPPER", None)
    bootstrap_env.pop("RUSTC_WORKSPACE_WRAPPER", None)
    bootstrap_env["RUSTFLAGS"] = ""
    bootstrap_env["CARGO_ENCODED_RUSTFLAGS"] = ""
    compiler = bootstrap_env.get("RUSTC") or bootstrap_env.get("CARGO_BUILD_RUSTC") or "rustc"
    host = next((line.removeprefix("host: ") for line in observer.probe([compiler, "-vV"]).splitlines() if line.startswith("host: ")), None)
    if not host:
        raise observer.ObservationError("helper bootstrap compiler does not report its host")
    bootstrap = subprocess.run([cargo, "build", "--release", "--manifest-path", str(source / "Cargo.toml"), "-p", "chelis-runtime-identity-build", "--target", host, "--target-dir", str(helper_target), "--message-format=json-render-diagnostics", "--quiet"], env=bootstrap_env, text=True, stdout=subprocess.PIPE, check=False, close_fds=False)
    helpers = set()
    for line in bootstrap.stdout.splitlines():
        event = json.loads(line)
        if event.get("reason") == "compiler-message" and event["message"].get("rendered"):
            sys.stderr.write(event["message"]["rendered"])
        if event.get("reason") == "compiler-artifact" and event["target"]["name"] == "chelis-runtime-identity-build" and event.get("executable"):
            helpers.add(event["executable"])
    if bootstrap.returncode:
        return bootstrap.returncode
    if len(helpers) != 1:
        raise observer.ObservationError("bootstrap did not report one exact host helper executable")
    helper = Path(helpers.pop())
    if environment.get("RUSTC_WORKSPACE_WRAPPER"):
        raise observer.ObservationError("RUSTC_WORKSPACE_WRAPPER cannot be silently hidden; unset it for managed identity production")
    if environment.get("RUSTC_WRAPPER") and environment["RUSTC_WRAPPER"] != environment.get("CHELIS_IDENTITY_HELPER"):
        environment["CHELIS_IDENTITY_INNER_WRAPPER"] = observer.transparent_wrapper(environment["RUSTC_WRAPPER"])
    environment.update({"RUSTC_WRAPPER": str(helper), "CHELIS_IDENTITY_HELPER": str(helper), "CHELIS_IDENTITY_PROTOCOL": "1", "CHELIS_IDENTITY_BACKEND": "cargo", "CHELIS_IDENTITY_WORKSPACE": str(workspace), "CHELIS_IDENTITY_STATE": str(state)})
    environment["CHELIS_IDENTITY_OBSERVED_WORKSPACE"] = str(workspace)
    if environment.get("CHELIS_IDENTITY_PROVENANCE") not in {"source-worktree", "sealed-distribution"}:
        raise observer.ObservationError("CHELIS_IDENTITY_PROVENANCE must explicitly select source-worktree or sealed-distribution")
    # A session-specific metadata file avoids cross-invocation metadata races.
    fd, metadata_path = tempfile.mkstemp(prefix="cargo-metadata-", suffix=".json", dir=state)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream)
    environment_path = metadata_path + ".environment"
    observer.atomic(environment_path, {name: observer.digest(os.fsencode(value)) for name, value in os.environ.items()})
    environment["CHELIS_IDENTITY_INVOCATION_ENVIRONMENT"] = environment_path
    environment["CHELIS_IDENTITY_METADATA"] = metadata_path
    environment["CHELIS_IDENTITY_SESSION"] = Path(metadata_path).stem
    launcher = install_cargo_launcher(state / "launchers" / environment["CHELIS_IDENTITY_SESSION"], real_cargo=cargo, python=sys.executable)
    environment["PATH"] = str(launcher.parent) + os.pathsep + environment.get("PATH", "")
    environment["CARGO"] = str(launcher)
    environment["CHELIS_IDENTITY_REAL_CARGO"] = cargo
    original_format = message_format(arguments)
    command_args = list(arguments)
    if original_format is None:
        separator = command_args.index("--") if "--" in command_args else len(command_args)
        command_args.insert(separator, "--message-format=json-render-diagnostics")
    elif not original_format.startswith("json"):
        raise observer.ObservationError("managed builds support Cargo JSON or default rendered diagnostics")
    failures = []
    producers = []
    proc = subprocess.Popen([cargo, *command_args], env=environment, stdout=subprocess.PIPE, close_fds=False)
    assert proc.stdout is not None
    try:
        for raw in iter(proc.stdout.readline, b""):
            try:
                event = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                sys.stdout.buffer.write(raw); sys.stdout.buffer.flush(); continue
            if not isinstance(event, dict):
                sys.stdout.buffer.write(raw); sys.stdout.buffer.flush(); continue
            if original_format is not None:
                sys.stdout.buffer.write(raw); sys.stdout.buffer.flush()
            elif event.get("reason") == "compiler-message":
                rendered = event.get("message", {}).get("rendered")
                if rendered:
                    sys.stderr.write(rendered); sys.stderr.flush()
            elif event.get("reason") not in {"compiler-artifact", "build-script-executed", "build-finished"}:
                sys.stdout.buffer.write(raw); sys.stdout.buffer.flush()
            try:
                if event.get("reason") == "compiler-artifact":
                    # Publish independently of receipt validity. Only the
                    # runtime closure interprets dependency observation errors.
                    for filename in event["filenames"]:
                        observer.atomic(state / "events" / environment["CHELIS_IDENTITY_SESSION"] / (observer.key(filename) + ".json"), event)
                    receipt_path = event_receipt(event, state)
                    producer_package = next((package["name"] for package in metadata["packages"] if package["id"] == event["package_id"]), "") in {"chelis-runtime", "chelis-cli", "chelis-python"}
                    if receipt_path:
                        try:
                            receipt = observer.load(receipt_path)
                        except observer.ObservationError:
                            if producer_package:
                                raise
                            continue
                        for output in receipt["outputs"]:
                            observer.atomic(state / "events" / environment["CHELIS_IDENTITY_SESSION"] / (observer.key(output["path"]) + ".json"), event)
                        if receipt.get("role"):
                            producers.append((receipt_path, event))
                    elif producer_package and not event["profile"].get("test") and "custom-build" not in event["target"]["kind"]:
                        raise observer.ObservationError("CHELIS_IDENTITY_MISSING_OBSERVATION: cached producer has no exact observation; rebuild in a clean managed target")
                elif event.get("reason") == "build-script-executed":
                    observer.atomic(state / "build-events" / environment["CHELIS_IDENTITY_SESSION"] / (observer.key(event["out_dir"]) + ".json"), event)
            except (OSError, ValueError, KeyError, observer.ObservationError) as error:
                failures.append(str(error))
        code = proc.wait()
        if code:
            return code
        # Fresh Cargo units require the same current source/output evidence as a
        # newly compiled one. Validation executes only after the event stream is
        # drained, so -j1 and concurrent rustc jobs cannot deadlock.
        for receipt_path, event in producers:
            verification = subprocess.run([str(helper), "validate-observation", receipt_path], env=environment, check=False)
            if verification.returncode:
                failures.append(f"stale/corrupt producer observation: {event['package_id']}")
        if failures:
            raise observer.ObservationError("; ".join(failures))
        return 0
    finally:
        if proc.poll() is None:
            proc.terminate(); proc.wait()
        os.unlink(metadata_path)
        os.unlink(environment_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cargo", default=os.environ.get("CHELIS_IDENTITY_REAL_CARGO"))
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    args = options.arguments
    if args[:1] == ["--"]:
        args = args[1:]
    cargo = options.cargo or shutil.which("cargo")
    if not cargo:
        parser.error("a real Cargo executable is required")
    if Path(cargo).resolve() == Path(__file__).resolve():
        parser.error("--cargo must not recurse into the identity driver")
    try:
        with open(cargo, "rb") as stream:
            prefix = stream.read(512)
        if b"chelis-runtime-identity-cargo-launcher" in prefix or (options.cargo is None and prefix.startswith(b"#!")):
            raise observer.ObservationError("PATH cargo is a launcher; --cargo or CHELIS_IDENTITY_REAL_CARGO must name actual Cargo")
        return run(cargo, args)
    except (OSError, KeyError, ValueError, observer.ObservationError) as error:
        print(f"runtime identity: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
