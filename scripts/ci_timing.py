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
    details = {"outcome": "success"}
    try:
        yield details
    except BaseException as error:
        details["outcome"] = type(error).__name__
        if isinstance(error, subprocess.CalledProcessError):
            details["returncode"] = error.returncode
        raise
    finally:
        record({"event": "finish", "name": name, "kind": kind,
                "seconds": time.monotonic() - start, **details})


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

    def run(*args, **kwargs):
        command = args[0] if args else kwargs.get("args", ())
        name = (os.fsdecode(command) if isinstance(command, (str, bytes, os.PathLike))
                else " ".join(map(str, command)))
        with span(name, "subprocess") as timing:
            completed = original(*args, **kwargs)
            timing["returncode"] = completed.returncode
            if completed.returncode:
                timing["outcome"] = "failure"
            return completed

    run._chelis_timed = True
    subprocess.run = run
    try:
        yield
    finally:
        subprocess.run = original
