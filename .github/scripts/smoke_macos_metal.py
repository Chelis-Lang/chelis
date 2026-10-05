#!/usr/bin/env python3
"""macOS smoke test for `chelis build --target metal`.

Drives two fixed-shape elementwise programs through the Metal backend
A Surf source and a span-attributed Deep source. The native build archives the
generated Objective-C++ code. A driver then links that library with clang++ to
prove that:

  1. `chelis build --target metal` produces a `.mm`, a header, and the
     Metal runtime header.
  2. `chelis build --target metal --deep` accepts span-attributed
     Deep input (S5 ingestion path) and the emitted `.mm` carries
     `// span:` comments derived from the input metadata.
  3. Apple's clang++ accepts the emitted Objective-C++ + embedded MSL
     kernel strings on both code paths.
  4. The Metal framework symbols resolve at link time.

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
import re
import subprocess
import sys
import tempfile
from pathlib import Path


SURF_PROGRAM = """\
def simple_add(a: tensor[8, f32], b: tensor[8, f32]) -> tensor[8, f32] = add(a, b)
"""

# Span-attributed Deep equivalent of the Surf program. Span IDs use
# distinct opaque strings so the post-build grep can lock the audit
# chain. Mirrors the shape of `simple_add(a, b) = add(a, b)` but
# routed through the Deep parser via `--deep`.
DEEP_PROGRAM = """\
(def {span: "wrap_simple_add"}
  simple_add
  (fn {}
    (params {}
      (a {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))})
      (b {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))}))
    (app {span: "src.add"}
      (var {} add)
      (var {span: "src.a"} a)
      (var {span: "src.b"} b))))
"""

# Minimum number of `// span:` lines we expect from the span-attributed
# Deep input. The current S4 emit path produces both source spans
# (`src.*`) AND optimization-pass synthesized markers
# (`__synthesized_tier2__` etc.) in non-trivial DAG shapes. We don't
# pin the exact count because optimizer changes can legitimately move
# it; we just lock that the count is non-zero.
MIN_DEEP_SPAN_COUNT = 1


_GENERATED_ENTRY_FUNCTION = re.compile(
    # `_mask_non_code` turns the exact three-character `"C"` linkage literal
    # into three NUL barriers while leaving the surrounding code searchable.
    r"extern\s+\x00{3}\s+void\s+(?P<entry_name>[A-Za-z_][A-Za-z0-9_]*)\s*\("
    r"\s*chelis_tensor\s*\*\*\s*inputs\s*,\s*int\s+n_in\s*,"
    r"\s*chelis_tensor\s*\*\*\s*outputs\s*,\s*int\s+n_out\s*\)\s*\{"
)
_GENERATED_ENTRY_END = re.compile(r"^}\s*$", re.MULTILINE)
_NON_CODE_TOKEN = re.compile(
    r"//[^\r\n]*|/\*.*?\*/|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'",
    re.DOTALL,
)
_HOST_PREPROCESSOR_DIRECTIVE = re.compile(
    r"^[ \t]*#[ \t]*(?P<name>[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
)
_GUARDED_OUTPUT_WRITEBACK = re.compile(
    r"outputs\[(?P<output_index>\d+)\]\s*=\s*chelis_alloc\([^;\n]+\);"
    r"(?:[ \t]*})?\s*"
    r"chelis_tensor_write\s+\*(?P<prefix>root|store)_guard_(?P=output_index)\s*=\s*"
    r"chelis_tensor_begin_write\(outputs\[(?P=output_index)\]\);\s*"
    r"chelis_write_view\s+(?P=prefix)_view_(?P=output_index)\s*=\s*"
    r"chelis_tensor_write_view\((?P=prefix)_guard_(?P=output_index)\);\s*"
    r"chelis_metal_device_to_host\((?P=prefix)_view_(?P=output_index)\.data\s*,"
    r"[^;\n]+\);\s*"
    r"chelis_tensor_end_write\((?P=prefix)_guard_(?P=output_index)\);"
)


def _mask_non_code(text: str) -> str:
    """Turn non-code tokens into offset-preserving barriers."""

    return _NON_CODE_TOKEN.sub(
        lambda token: "".join(
            "\n" if character == "\n" else "\0" for character in token.group(0)
        ),
        text,
    )


def guarded_output_writeback_indices(mm_text: str, entry_name: str) -> set[int]:
    """Return outputs the named entry initializes through an opaque write lease.

    A device-to-host call alone is not evidence that an output was initialized.
    The allocation, begin-write, matching write-view, transfer, and matching
    end-write must be one contiguous generated sequence inside one exported
    tensor entry-function body whose symbol is ``entry_name``. Matching the
    generated root/store stem and numeric suffix binds all five operations to
    the same output without allowing unrelated functions to contribute
    individual operations or a complete proof for a different exported symbol.
    """

    initialized: set[int] = set()
    code_text = _mask_non_code(mm_text)
    if any(
        directive.group("name") not in {"include", "import"}
        for directive in _HOST_PREPROCESSOR_DIRECTIVE.finditer(code_text)
    ):
        # Generated host code has no conditional or macro-definition surface.
        # Refuse rather than accepting writeback evidence disabled or synthesized
        # by the preprocessor. Directives in embedded MSL literals were masked.
        return initialized
    for entry in _GENERATED_ENTRY_FUNCTION.finditer(code_text):
        if entry.group("entry_name") != entry_name:
            continue
        entry_end = _GENERATED_ENTRY_END.search(code_text, entry.end())
        if entry_end is None:
            continue
        body = code_text[entry.end() : entry_end.start()]
        initialized.update(
            int(writeback.group("output_index"))
            for writeback in _GUARDED_OUTPUT_WRITEBACK.finditer(body)
        )
    return initialized


def run(cmd, **kwargs):
    """Run `cmd`, echoing it, and check exit status."""
    print("+ " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def smoke_one(
    chelis: Path,
    tmpdir: Path,
    *,
    label: str,
    src_name: str,
    program: str,
    extra_args: list[str],
    expect_spans: bool,
) -> int:
    """Run one Metal smoke iteration. Returns 0 on success, non-zero
    on the first failure (mirroring the original single-case shape so
    callers can short-circuit on the first failure)."""
    src = tmpdir / src_name
    out_dir = tmpdir / f"metal-output-{label}"
    out_dir.mkdir()
    src.write_text(program)

    run(
        [
            str(chelis),
            "build",
            str(src),
            *extra_args,
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
                f"smoke_macos_metal[{label}]: expected output {required} missing",
                file=sys.stderr,
            )
            return 3

    mm_text = mm_path.read_text()
    if "M1 fallback stub" in mm_text:
        print(
            f"smoke_macos_metal[{label}]: emitted .mm is the M1 stub (M2 emission "
            "did not handle this DAG)",
            file=sys.stderr,
        )
        return 4
    for marker in ("kernel void", "[[thread_position_in_grid]]", "chelis_metal_launch"):
        if marker not in mm_text:
            print(
                f"smoke_macos_metal[{label}]: emitted .mm missing required MSL marker: {marker!r}",
                file=sys.stderr,
            )
            return 4
    if 0 not in guarded_output_writeback_indices(mm_text, "simple_add"):
        print(
            f"smoke_macos_metal[{label}]: emitted .mm does not initialize outputs[0] "
            "through allocation + begin-write + matching write-view + device copy + "
            "matching end-write in order",
            file=sys.stderr,
        )
        return 4

    if expect_spans:
        # S5.3 audit-chain assertion: span-attributed Deep input must
        # produce `// span:` lines in the emitted .mm. Closes the
        # macOS-side audit-chain loop for the Metal backend.
        span_count = mm_text.count("// span:")
        if span_count < MIN_DEEP_SPAN_COUNT:
            print(
                f"smoke_macos_metal[{label}]: span-attributed Deep input "
                f"produced {span_count} `// span:` lines in emitted .mm "
                f"(expected at least {MIN_DEEP_SPAN_COUNT}). The audit "
                "chain on macOS Metal is broken or codegen has regressed.",
                file=sys.stderr,
            )
            return 6

    driver = tmpdir / f"driver_{label}.mm"
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

    bin_path = tmpdir / f"metal_smoke_bin_{label}"
    run(
        [
            "xcrun",
            "-sdk",
            "macosx",
            "clang++",
            "-std=c++17",
            "-fobjc-arc",
            "-O2",
            str(driver),
            str(out_dir / "libsimple_add_metal.a"),
            f"-I{out_dir}",
            str(out_dir / "libchelis_runtime.a"),
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-o",
            str(bin_path),
        ],
        timeout=120,
    )

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
            f"smoke_macos_metal[{label}]: nm output does not reference any Metal "
            "symbol; compile-and-link likely fell through to a no-op",
            file=sys.stderr,
        )
        print(nm.stdout, file=sys.stderr)
        return 5

    print(
        f"smoke_macos_metal[{label}]: compile + link succeeded; Metal symbols resolved"
        + (" (with span comments)" if expect_spans else "")
    )
    return 0


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

        # Surf path (existing M-phase smoke).
        rc = smoke_one(
            chelis,
            tmpdir,
            label="surf",
            src_name="simple_add.ch",
            program=SURF_PROGRAM,
            extra_args=[],
            expect_spans=False,
        )
        if rc != 0:
            return rc

        # Deep path (S5.3): span-attributed Deep input through `--deep`.
        rc = smoke_one(
            chelis,
            tmpdir,
            label="deep",
            src_name="simple_add.dp",
            program=DEEP_PROGRAM,
            extra_args=["--deep"],
            expect_spans=True,
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
