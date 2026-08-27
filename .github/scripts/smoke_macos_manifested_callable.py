#!/usr/bin/env python3
"""macOS compile-link-run smoke for the manifested C-callable ABI.

Drives a parameterized fixed-shape f32 elementwise addition through the
default C target, links the generated callable with a driver, and compares
the result bits exactly. The callable has no automatic root: its two tensor
inputs are supplied by the driver and its sole result occupies outputs[0].

The superseded self-bound source remains an executable [05-UNS-1] negative
control. It must fail before creating an artifact because `chelis build` has
no runtime values for the automatic roots it owes.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


ADD_PROGRAM = (
    "def add_kernel(a: tensor[2, 4, f32], b: tensor[2, 4, f32]) "
    "-> tensor[2, 4, f32] = (add(a, b) : tensor[2, 4, f32])\n"
)

SELF_BOUND_PROGRAM = """\
a = (a : tensor[2, 3, f32])
b = (b : tensor[3, 4, f32])
out = (matmul(a, b) : tensor[2, 4, f32])
"""

DRIVER_C = """\
#include "chelis_runtime.h"
#include <stdint.h>
#include <stdio.h>
#include <string.h>

void manifested_callable(
    chelis_tensor **inputs,
    int n_in,
    chelis_tensor **outputs,
    int n_out
);

int main(void) {
    int64_t shape[2] = {2, 4};
    chelis_tensor *a = chelis_alloc(2, shape, CHELIS_DTYPE_F32);
    chelis_tensor *b = chelis_alloc(2, shape, CHELIS_DTYPE_F32);
    const uint32_t a_bits[8] = {
        0x3f800000u, 0x3f800001u, 0xbf800000u, 0x00000001u,
        0x7f7fffffu, 0x00800000u, 0x80000000u, 0x3eaaaaabu
    };
    const uint32_t b_bits[8] = {
        0x33800000u, 0x33800000u, 0x3f800000u, 0x00000001u,
        0xff7fffffu, 0x80800000u, 0x00000000u, 0x3eaaaaabu
    };
    const uint32_t expected_bits[8] = {
        0x3f800000u, 0x3f800002u, 0x00000000u, 0x00000002u,
        0x00000000u, 0x00000000u, 0x00000000u, 0x3f2aaaabu
    };
    chelis_tensor *inputs[2] = {a, b};
    chelis_tensor *outputs[1] = {0};

    memcpy(a->data, a_bits, sizeof(a_bits));
    memcpy(b->data, b_bits, sizeof(b_bits));
    manifested_callable(inputs, 2, outputs, 1);

    if (outputs[0] == NULL) {
        fprintf(stderr, "manifested callable did not populate outputs[0]\\n");
        return 1;
    }
    if (outputs[0]->dtype != CHELIS_DTYPE_F32 || outputs[0]->rank != 2 ||
        outputs[0]->shape == NULL || outputs[0]->strides == NULL ||
        outputs[0]->shape[0] != 2 || outputs[0]->shape[1] != 4 ||
        outputs[0]->strides[0] != 4 || outputs[0]->strides[1] != 1 ||
        outputs[0]->size != 8 || outputs[0]->byte_capacity < 32 ||
        outputs[0]->owns_data != 1 || outputs[0]->reserved[0] != 0 ||
        outputs[0]->reserved[1] != 0) {
        fprintf(stderr, "manifested callable returned the wrong output ABI\\n");
        return 1;
    }
    if (memcmp(outputs[0]->data, expected_bits, sizeof(expected_bits)) != 0) {
        uint32_t got_bits[8];
        memcpy(got_bits, outputs[0]->data, sizeof(got_bits));
        for (int i = 0; i < 8; ++i) {
            if (got_bits[i] != expected_bits[i]) {
                fprintf(
                    stderr,
                    "element %d: got 0x%08x expected 0x%08x\\n",
                    i,
                    got_bits[i],
                    expected_bits[i]
                );
            }
        }
        return 1;
    }

    chelis_free(outputs[0]);
    chelis_free(a);
    chelis_free(b);
    return 0;
}
"""


def run(cmd: list[object], *, timeout: int) -> None:
    """Run ``cmd``, echo it, and require a successful exit status."""

    rendered = [str(part) for part in cmd]
    print("+ " + " ".join(rendered))
    try:
        subprocess.run(rendered, check=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(
            f"command timed out after {timeout}s: " + " ".join(rendered),
            file=sys.stderr,
        )
        sys.exit(124)


def require_unavailable_root(
    chelis: Path, source: Path, output_dir: Path, *, timeout: int
) -> None:
    """Require the self-bound program to fail through [05-UNS-1]."""

    cmd = [str(chelis), "build", str(source), "--output", str(output_dir)]
    print("+ " + " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(
            f"command timed out after {timeout}s: " + " ".join(cmd),
            file=sys.stderr,
        )
        sys.exit(124)

    expected = "[05-UNS-1] unavailable root `a`"
    if result.returncode == 0 or expected not in result.stderr:
        print(
            "self-bound negative control did not produce the required "
            f"diagnostic {expected!r}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            file=sys.stderr,
        )
        sys.exit(3)
    if output_dir.exists():
        print(
            "self-bound negative control left a partial output artifact at "
            f"{output_dir}",
            file=sys.stderr,
        )
        sys.exit(3)


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
            f"smoke_macos_manifested_callable: chelis binary not found at {chelis}",
            file=sys.stderr,
        )
        return 2

    with tempfile.TemporaryDirectory(
        prefix="chelis_manifested_callable_smoke_"
    ) as tmp:
        tmpdir = Path(tmp)

        negative_source = tmpdir / "self_bound.ch"
        negative_output = tmpdir / "self_bound_out"
        negative_source.write_text(SELF_BOUND_PROGRAM, encoding="utf-8")
        require_unavailable_root(
            chelis,
            negative_source,
            negative_output,
            timeout=30,
        )

        source = tmpdir / "manifested_callable.ch"
        output_dir = tmpdir / "out"
        source.write_text(ADD_PROGRAM, encoding="utf-8")

        print("Generating manifested C callable")
        run([chelis, "build", source, "--output", output_dir], timeout=30)

        driver = output_dir / "driver.c"
        driver.write_text(DRIVER_C, encoding="utf-8")

        callable_o = output_dir / "manifested_callable.o"
        driver_o = output_dir / "driver.o"
        runtime_a = output_dir / "libchelis_runtime.a"
        bin_path = output_dir / "manifested_callable_bin"

        print("Compiling manifested callable and driver")
        run(
            [
                "clang",
                "-O2",
                "-I",
                output_dir,
                "-c",
                output_dir / "manifested_callable.c",
                "-o",
                callable_o,
            ],
            timeout=30,
        )
        run(
            [
                "clang",
                "-O2",
                "-I",
                output_dir,
                "-c",
                driver,
                "-o",
                driver_o,
            ],
            timeout=30,
        )

        print("Linking manifested callable smoke binary")
        run(
            ["clang", callable_o, driver_o, runtime_a, "-o", bin_path],
            timeout=30,
        )
        print("Running manifested callable smoke binary")
        run([bin_path], timeout=30)
        print("macOS manifested C-callable smoke passed")

    return 0


if __name__ == "__main__":
    sys.exit(main())
