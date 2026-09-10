"""Diagnostic timings only; never supply correctness or cache authority."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import time


def record(row):
    directory = os.environ.get("CHELIS_CI_TIMING_DIR")
    if directory:
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / f"{os.getpid()}.jsonl").open("a") as output:
            output.write(json.dumps({"pid": os.getpid(), **row}) + "\n")


@contextmanager
def span(name, kind):
    start = time.monotonic()
    record({"event": "start", "name": name, "kind": kind})
    outcome = "success"
    try:
        yield
    except BaseException as error:
        outcome = type(error).__name__
        raise
    finally:
        record({"event": "finish", "name": name, "kind": kind,
                "seconds": time.monotonic() - start, "outcome": outcome})


@contextmanager
def subprocesses():
    """Instrument this process's run calls without changing their arguments/results.

    Nested Python entry points opt in independently. No environment values or
    captured child output are logged. Start records survive job cancellation.
    """
    if (not os.environ.get("CHELIS_CI_TIMING_DIR")
            or getattr(subprocess.run, "_chelis_timed", None) is True):
        yield
        return
    original = subprocess.run

    def run(command, *args, **kwargs):
        name = command if isinstance(command, str) else " ".join(map(str, command))
        with span(name, "subprocess"):
            return original(command, *args, **kwargs)

    run._chelis_timed = True
    subprocess.run = run
    try:
        yield
    finally:
        subprocess.run = original
