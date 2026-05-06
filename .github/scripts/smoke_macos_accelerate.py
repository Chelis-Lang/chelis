#!/usr/bin/env python3
"""macOS smoke test for `chelis build` (default C target) on Apple's
Accelerate BLAS path.

Drives a fixed-shape matmul through the C backend, links the generated
C against macOS's Accelerate framework, and runs the resulting binary.
Asserts that:

  1. `chelis build` produces a `.c` file whose BLAS surface is the
     Accelerate-compatible `cblas_sgemm` plus a `chelis_blas.h` include.
  2. clang accepts the emitted C, links it against `-framework Accelerate`,
     and the resulting matmul binary returns a numerically correct result.

Behaviorally identical to the previous shell-based runner. Ported per
the project-wide "Never shell" policy (CLAUDE.md Scripting Language Policy
+ spec/01-nomenclature.md §2.9).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


MATMUL_PROGRAM = """\
a = (a : tensor[2, 3, f32])
b = (b : tensor[3, 4, f32])
out = (matmul(a, b) : tensor[2, 4, f32])
"""

DRIVER_C = """\
#include "chelis_runtime.h"
#include <math.h>
#include <stdio.h>

void matmul(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);

int main(void) {
    int a_shape[2] = {2, 3};
    int b_shape[2] = {3, 4};
    chelis_tensor *a = chelis_alloc(2, a_shape, CHELIS_F32);
    chelis_tensor *b = chelis_alloc(2, b_shape, CHELIS_F32);
    float a_values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float b_values[12] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    };
    float expected[8] = {
        38.0f, 44.0f, 50.0f, 56.0f,
        83.0f, 98.0f, 113.0f, 128.0f
    };
    chelis_tensor *inputs[2] = {a, b};
    chelis_tensor *outputs[3] = {0};

    for (int i = 0; i < 6; ++i) {
        a->data[i] = a_values[i];
    }
    for (int i = 0; i < 12; ++i) {
        b->data[i] = b_values[i];
    }

    matmul(inputs, 2, outputs, 3);

    if (outputs[2] == NULL) {
        fprintf(stderr, "matmul did not populate the result output\\n");
        return 1;
    }
    for (int i = 0; i < 8; ++i) {
        if (fabsf(outputs[2]->data[i] - expected[i]) > 1e-4f) {
            fprintf(stderr, "matmul mismatch at %d: got %f expected %f\\n", i, outputs[2]->data[i], expected[i]);
            return 1;
        }
    }

    for (int i = 0; i < 3; ++i) {
        chelis_free(outputs[i]);
    }
    chelis_free(a);
    chelis_free(b);
    return 0;
}
"""


def run(cmd, *, timeout: int) -> None:
    """Run `cmd`, echoing it, and check exit status. Mirrors the
    `run_with_timeout` helper in the prior shell version."""
    print("+ " + " ".join(str(c) for c in cmd))
    try:
        subprocess.run([str(c) for c in cmd], check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(
            f"command timed out after {timeout}s: "
            + " ".join(str(c) for c in cmd),
            file=sys.stderr,
        )
        sys.exit(124)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chelis",
        type=Path,
        help="path to the `chelis` binary (e.g. ./target/debug/chelis)",
    )
    args = parser.parse_args()

    chelis = args.chelis.resolve()
    if not chelis.exists():
        print(
            f"smoke_macos_accelerate: chelis binary not found at {chelis}",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(prefix="chelis_accelerate_smoke_") as tmp:
        tmpdir = Path(tmp)
        src = tmpdir / "matmul.ch"
        out_dir = tmpdir / "out"
        src.write_text(MATMUL_PROGRAM)

        print("Generating C with chelis")
        run(
            [chelis, "build", src, "--output", out_dir],
            timeout=30,
        )

        print("Checking generated BLAS surface")
        emitted_c = (out_dir / "matmul.c").read_text()
        for marker in ("cblas_sgemm", "chelis_blas.h"):
            if marker not in emitted_c:
                print(
                    f"smoke_macos_accelerate: emitted matmul.c missing "
                    f"required BLAS surface marker: {marker!r}",
                    file=sys.stderr,
                )
                return 3

        driver = out_dir / "driver.c"
        driver.write_text(DRIVER_C)

        matmul_o = out_dir / "matmul.o"
        driver_o = out_dir / "driver.o"
        runtime_a = out_dir / "libchelis_runtime.a"
        bin_path = out_dir / "matmul_bin"

        print("Compiling generated C")
        run(
            [
                "clang",
                "-O2",
                "-I",
                out_dir,
                "-c",
                out_dir / "matmul.c",
                "-o",
                matmul_o,
            ],
            timeout=30,
        )
        run(
            [
                "clang",
                "-O2",
                "-I",
                out_dir,
                "-c",
                driver,
                "-o",
                driver_o,
            ],
            timeout=30,
        )
        print("Linking against Accelerate")
        run(
            [
                "clang",
                matmul_o,
                driver_o,
                runtime_a,
                "-framework",
                "Accelerate",
                "-o",
                bin_path,
            ],
            timeout=30,
        )
        print("Running matmul smoke binary")
        run([bin_path], timeout=30)
        print("macOS Accelerate smoke test passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
