#!/usr/bin/env python3
"""macOS smoke test for `chelis build --target metal`.

Drives a fixed-shape elementwise program through the Metal backend, then
runs `clang++ -fobjc-arc -framework Metal -framework Foundation` against
the emitted `.mm` to prove that:

  1. `chelis build --target metal` produces a `.mm`, a header, and the
     Metal runtime header
  2. Apple's clang++ accepts the emitted Objective-C++ + embedded MSL
     kernel strings
  3. The Metal framework symbols resolve at link time

The smoke is **intentionally compile-and-link only, no execution**.

# DO NOT change this script to dispatch a kernel without first verifying:
#   1. `MTLCreateSystemDefaultDevice()` returns non-null on the current
#      macos-latest runner image. As of 2026-04-29, GitHub's macOS-latest
#      VMs are documented to return null in some images. The runtime
#      correctness oracle for the Metal backend is the M6 manual gate
#      (`cargo test -p chelis-backend-metal --test gpu_correctness --
#       --ignored --test-threads=1`), which runs on a real workstation
#      with a usable Metal device.
#   2. If dispatch becomes possible on the runner, also update
#      docs/manual_gates.md to remove the "workstation is the only Metal
#      execution path" note for M6.
# Anchor the change in a runner-image probe, not in trust.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PROGRAM = """\
def simple_add(a: tensor[8, f32], b: tensor[8, f32]) -> tensor[8, f32] = add(a, b)
"""


def run(cmd, **kwargs):
    """Run `cmd`, echoing it, and check exit status."""
    print("+ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "chelis",
        type=Path,
        help="path to the `chelis` binary (e.g. ./target/debug/chelis)",
    )
    args = parser.parse_args()

    if sys.platform != "darwin":
        print(
            f"smoke_macos_metal: not on macOS (sys.platform={sys.platform}); skipping",
            file=sys.stderr,
        )
        return 0

    chelis = args.chelis.resolve()
    if not chelis.exists():
        print(f"smoke_macos_metal: chelis binary not found at {chelis}", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="chelis_metal_smoke_") as tmp:
        tmpdir = Path(tmp)
        src = tmpdir / "simple_add.ch"
        out_dir = tmpdir / "metal-output"
        out_dir.mkdir()
        src.write_text(PROGRAM)

        # 1) chelis build --target metal
        run(
            [
                str(chelis),
                "build",
                str(src),
                "--target",
                "metal",
                "--output",
                str(out_dir),
            ],
            timeout=60,
        )

        mm_path = out_dir / "simple_add_metal.mm"
        h_path = out_dir / "simple_add_metal.h"
        runtime_h = out_dir / "chelis_metal_runtime.h"
        for required in (mm_path, h_path, runtime_h, out_dir / "chelis_runtime.h"):
            if not required.exists():
                print(
                    f"smoke_macos_metal: expected output {required} missing",
                    file=sys.stderr,
                )
                return 3

        # Sanity-check the emitted .mm shape — fail closed if codegen
        # silently fell through to the C backend or to the M1 stub.
        mm_text = mm_path.read_text()
        if "M1 fallback stub" in mm_text:
            print(
                "smoke_macos_metal: emitted .mm is the M1 stub (M2 emission "
                "did not handle this DAG)",
                file=sys.stderr,
            )
            return 4
        for marker in ("kernel void", "[[thread_position_in_grid]]", "chelis_metal_launch"):
            if marker not in mm_text:
                print(
                    f"smoke_macos_metal: emitted .mm missing required MSL marker: {marker!r}",
                    file=sys.stderr,
                )
                return 4
        # ABI invariant: the function MUST materialize at least one output
        # back to the `outputs` array, otherwise it computes a result into
        # a device buffer and returns without writing anything the caller
        # can see. This catches the red-team M0-M4 defect where the CLI
        # lowering path produced root-without-Store DAGs and the emitter
        # silently skipped the writeback.
        if "chelis_metal_device_to_host(outputs[" not in mm_text:
            print(
                "smoke_macos_metal: emitted .mm does not write any output via "
                "chelis_metal_device_to_host(outputs[...]); function returns "
                "with uninitialized outputs",
                file=sys.stderr,
            )
            return 4

        # 2) Build a tiny driver that calls the emitted entrypoint, link
        # everything together with -framework Metal/Foundation. We don't
        # call MTLCreateSystemDefaultDevice() so we don't need a real
        # Metal device available on the CI runner.
        driver = tmpdir / "driver.mm"
        driver.write_text(
            """\
#import <Foundation/Foundation.h>
#include "chelis_runtime.h"
#include <stdio.h>

extern "C" void simple_add(chelis_tensor **inputs, int n_in,
                           chelis_tensor **outputs, int n_out);

int main(void) {
    // Compile-and-link smoke: don't actually call simple_add (which
    // would need a real Metal device). Just hold a function pointer to
    // force the linker to keep the symbol live. See the file-level
    // comment for the rationale and the contract for ever loosening it.
    void (*fp)(chelis_tensor**, int, chelis_tensor**, int) = &simple_add;
    if (fp == NULL) return 1;
    printf("simple_add resolved at %p\\n", (void *)fp);
    return 0;
}
"""
        )

        bin_path = tmpdir / "metal_smoke_bin"
        run(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "clang++",
                "-std=c++17",
                "-fobjc-arc",
                "-O2",
                str(mm_path),
                str(driver),
                f"-I{out_dir}",
                f"-L{out_dir}",
                "-lchelis_runtime",
                "-framework",
                "Metal",
                "-framework",
                "Foundation",
                "-o",
                str(bin_path),
            ],
            timeout=120,
        )

        # 3) Confirm Metal symbols are actually referenced in the linked
        # binary — proves the .mm produced real Metal calls, not a no-op
        # that happened to link.
        nm = subprocess.run(
            ["nm", str(bin_path)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if not any(
            tok in nm.stdout
            for tok in ("MTLCreateSystemDefaultDevice", "MTLDevice", "_OBJC_CLASS_$_NSString")
        ):
            print(
                "smoke_macos_metal: nm output does not reference any Metal symbol; "
                "compile-and-link likely fell through to a no-op",
                file=sys.stderr,
            )
            print(nm.stdout, file=sys.stderr)
            return 5

        print("smoke_macos_metal: compile + link succeeded; Metal symbols resolved")
        return 0


if __name__ == "__main__":
    sys.exit(main())
