# Local HIP Environment

The HIP manual gates need an AMD GPU and a ROCm toolchain. This runbook describes the
supported local configuration: ROCm installed from AMD's Python wheels, which
`scripts/hip_test.py` expects. Not every developer machine has that GPU or toolchain,
and nothing here implies one does.

## 1. Quick reference (run HIP tests this way)

Use `scripts/hip_test.py` — it sets the full hipBLAS env from the wheel paths of §3,
verifies the paths exist, and execs `cargo test` with your args:

```sh
scripts/hip_test.py -p chelis-backend-hip --test gpu_correctness -- --ignored --test-threads=1
scripts/hip_test.py -p chelis-cli --test cross_library_semantic_gap_hip_gpu -- --ignored --test-threads=1
# Inside an active Devenv shell, replace scripts/hip_test.py with chelis-hip-test.
```

The wrapper is required for any test that links `libhipblas`. Plain
`cargo test --ignored ...` inherits only the systemd `environment.d/hip.conf`
defaults of §3.1 — the `-isystem` half of `HIPCC_COMPILE_FLAGS_APPEND` and
`HSA_OVERRIDE_GFX_VERSION=11.0.0`. That is enough for non-hipBLAS HIP
tests, but for hipBLAS-linked tests the binary segfaults at process exit
with empty stdout/stderr — the failure mode looks like a code regression
but is purely environmental. The Rust test harness emits an explicit hint
when it detects this pattern.

If you need the env in your own shell:

```sh
eval $(scripts/hip_test.py --print-env)
```

The sections below explain why each piece is necessary; read them when
diagnosing a workstation breakage, not on the happy path.

## 2. Toolchain layout

- `rocminfo` is the source of truth for local GPU probing. `rocm-smi` may be
  absent; do not assume it exists in instructions or validation scripts.
- The `_rocm_sdk_core` Python wheel is the single authoritative HIP stack. The
  wheel ships its own headers, clang, device libraries, and
  `rocminfo`/`rocm-smi`/`hipconfig` binaries, all under
  `~/.local/lib/python3.12/site-packages/_rocm_sdk_core/`; its `hipcc` is the one on
  PATH.
- The device library package is the `_rocm_sdk_libraries_gfx1151` wheel. On a host
  where `rocminfo` reports the device as `gfx1100`, the wheel stack still expects the
  gfx1151 library lane, which `HSA_OVERRIDE_GFX_VERSION=11.5.1` selects (§3).

A distribution HIP development package (for example Fedora's `rocm-hip-devel`)
installed alongside the wheel ships older, conflicting headers under
`/usr/include/hip/`. Without intervention, the wheel's clang resolves
`<hip/hip_runtime.h>` against those system headers via clang's standard
`/usr/include` search path, because the wheel's own HIP include is added with
`-idirafter`, at lowest priority. Older headers reference an
`__AMDGCN_WAVEFRONT_SIZE` macro that the wheel's clang does not define:

```text
/usr/include/hip/amd_detail/amd_warp_functions.h:96:37: error: use of
undeclared identifier '__AMDGCN_WAVEFRONT_SIZE'
```

## 3. hipBLAS environment

For hipBLAS gates, run with the env `scripts/hip_test.py` sets, where `$HOME` is the
account that owns the wheels:

```sh
HSA_OVERRIDE_GFX_VERSION=11.5.1
LD_LIBRARY_PATH=$HOME/.local/lib/python3.12/site-packages/_rocm_sdk_core/lib:$HOME/.local/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1151/lib
HIPCC_COMPILE_FLAGS_APPEND="-isystem $HOME/.local/lib/python3.12/site-packages/_rocm_sdk_core/include -L$HOME/.local/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1151/lib"
```

If the wheels live elsewhere, update `WHEEL_CORE` and `WHEEL_GFX` in
`scripts/hip_test.py`. Each piece is explained below.

### 3.1 Wheel-include override

Make the wheel's HIP include win over system headers durably, through the
systemd-user `environment.d` mechanism. Put the override in
`~/.config/environment.d/hip.conf`:

```text
HIPCC_COMPILE_FLAGS_APPEND=-isystem <home>/.local/lib/python3.12/site-packages/_rocm_sdk_core/include
```

with `<home>` replaced by the account's absolute home directory. systemd's user manager reads
`environment.d/*.conf` at user-session start and propagates the variable to every
process the user manager spawns, including login shells, graphical sessions,
non-interactive `bash -c` invocations, `cargo test --workspace`, and orchestrated or
CI-style harnesses. `hipcc` honors that variable and prepends the wheel HIP include as
a high-priority `-isystem` path, so `<hip/...>` resolves inside the wheel and the
system headers are ignored.

Setting the variable only in `~/.bashrc` is not enough: bash sources it only for
interactive shells, so non-interactive contexts never see it. A `~/.bashrc` line may
be kept for interactive shells that bypass systemd, but it must not conflict with the
`environment.d` value.

Removing the system HIP development package also works but requires root; the
env-var route is purely user-space, survives system updates, and only affects `hipcc`
invocations.

### 3.2 Unversioned `.so` symlinks

The core wheel ships only versioned `libamdhip64.so.7`, `libhiprtc.so.7`, and
`libhiprtc-builtins.so.7`, with no unversioned `.so` symlinks, so `hipcc -lamdhip64
-lhiprtc`, used by the Chelis HIP smoke tests, fails at link time. Add the unversioned
symlinks under `~/.local/lib/python3.12/site-packages/_rocm_sdk_core/lib/`.

The `_rocm_sdk_libraries_gfx1151` wheel ships `libhipblas.so.3`; add a user-space
`libhipblas.so -> libhipblas.so.3` symlink in its `lib/` directory so
`hipcc -lhipblas` resolves to the wheel library when the `-L` path above is present.
`scripts/hip_test.py` checks for that symlink. Recreate every symlink after a wheel
upgrade.

### 3.3 hipBLAS / rocBLAS compatibility

HIP gates that exercise hipBLAS require a single coherent ROCm stack at compile
time and runtime. Mixed stacks, such as compiling against one ROCm install's headers
while linking another install's hipBLAS, are unsupported. The gfx1151 wheel's rocBLAS
Tensile data lives under
`~/.local/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx1151/lib/rocblas/library/gfx1151`.

The descriptor/owner path selects hipBLAS major 3 or newer, with the
official `hipblas/hipblas.h` and its matching hipBLAS library. The version
header exposes `hipblasVersionMajor`; generated helpers use `hipDataType`
(`HIP_R_16F`, `HIP_R_16BF`) and `hipblasComputeType_t`
(`HIPBLAS_COMPUTE_32F`) for the corresponding GEMM calls. The official
[version-header template](https://raw.githubusercontent.com/ROCm/hipBLAS/develop/library/include/hipblas-version.h.in)
defines that version macro. A later major must retain the required signatures
or fail compilation; a major number alone supplies no execution evidence. Missing SDK
headers are a prerequisite failure; handwritten declarations inside Chelis are not a
supported substitute. The shared descriptor adoption under #893 removes the existing
support header's legacy declaration fallback and compiles helpers against the
selected official API.

Do not infer that API from an exported symbol name. AMD documents that hipBLAS
3.0 replaced `hipblasDatatype_t` with `hipDataType`, using
`hipblasComputeType_t` for GEMM's computation type
([hipBLAS deprecations](https://rocm.docs.amd.com/projects/hipBLAS/en/latest/reference/deprecation.html#removed-in-hipblas-3-0)).
A missing `hipblasGemmEx_v2` symbol does not prove that the unsuffixed symbol
accepts the older signature. This SDK choice does not change Chelis's dtype
or accumulator semantics.

The wheel paths in this runbook are an environment configuration, not an exact-head
header/library compatibility receipt. Accepting a change to the descriptor adoption
requires recording the installed header path and version, the preprocessed
declarations used by the actual generated translation unit, the linked library
path/version, and executed f32/f64/f16/bf16 matmul results. A fixture-only header
census cannot replace this hardware evidence.

Chelis does not add custom library-search environment variables or silently fall back
when hipBLAS is unavailable. Use standard ROCm and platform mechanisms
(`HSA_OVERRIDE_GFX_VERSION`, `LD_LIBRARY_PATH`, `HIPCC_COMPILE_FLAGS_APPEND`,
`ldconfig`, or a package-manager ROCm install) to make the intended libraries and
ISA lane visible.

## 4. Verification

Verify the durable fix from a fresh non-interactive shell after login, from the root
of a Chelis checkout:

```sh
bash -c 'env | grep HIPCC_COMPILE_FLAGS_APPEND'
bash -c 'cargo test -p chelis-backend-hip --test codegen_structure s14_generated_hip_source_compiles_when_hipcc_available'
```

Expected:

```text
HIPCC_COMPILE_FLAGS_APPEND=-isystem <home>/.local/lib/python3.12/site-packages/_rocm_sdk_core/include
... test s14_generated_hip_source_compiles_when_hipcc_available ... ok
```

`s14_generated_hip_source_compiles_when_hipcc_available`, in
`chelis-backend-hip/tests/codegen_structure.rs`, passes from non-interactive bash once
`~/.config/environment.d/hip.conf` is present and the user session has been
re-established by login or `systemctl --user import-environment`.

Also verify resolved HIP includes:

```sh
hipcc -E foo.hip | grep amd_warp_functions.h
```

The resolved path must be the wheel one. A trivial kernel must compile:

```sh
cat > trivial_kernel.hip <<'EOF'
#include <hip/hip_runtime.h>
__global__ void noop() {}
EOF
hipcc -c trivial_kernel.hip -o trivial_kernel.o
echo $?
```

Expected exit status: `0`.

## 5. Runtime header caveat

Any HIP source that does not `#include <hip/hip_runtime.h>` fails because the wheel
clang's auto-included `__clang_hip_runtime_wrapper.h` does not pull in the launch
helpers. This is upstream HIP behavior, not a workstation defect. Every kernel must
include the runtime header explicitly.

## 6. Ownership hardware gate in CI

`.github/workflows/ownership-hip.yml` exposes the compiled-value ownership hardware
gate as a manual GitHub Actions job. Default pull-request CI runs the ownership and
launch oracles that need no GPU; it does not execute this hardware gate.

The job runs on a Linux x64 self-hosted runner with the additional label
`chelis-hip-gfx1151`. The runner account must have the ROCm wheel environment
documented above, working GPU device permissions, `hipcc` and `rocminfo` on
PATH, and the normal C build prerequisites (C compiler, Clang, OpenBLAS,
ASan/UBSan). The two hardware fixtures invoke `scripts/hip_test.py`, so the
wheel paths must exist under that account's home directory. The label names
this specific configured environment; it does not make an arbitrary AMD host
compatible with the wheel wrapper. Runner registration and provisioning are
separate from the workflow. A job without a matching online runner
waits in the queue and supplies no acceptance evidence.

A maintainer dispatches it for a reviewed commit in this repository:

```text
gh workflow run ownership-hip.yml --ref main -f commit=<full-40-character-SHA>
```

GitHub requires a manually dispatched workflow to exist on the default branch
([workflow dispatch documentation](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow)).
The job has read-only repository access and never runs automatically for PRs.
Its `.yml` file deliberately uses JSON syntax (a YAML subset), so the Python
guard validates decoded event keys and rejects duplicate keys or other spellings.
Use a dedicated runner for trusted repository commits, without workstation
credentials or unrelated jobs sharing its GPU/build directory.

The same CI entry point works directly on a configured workstation or in
another CI service after checking out that exact commit and provisioning the
checkout's managed Python:

```text
.venv/bin/python scripts/ownership_hip_ci.py --expected-head <full-40-character-SHA>
```

It runs `.venv/bin/python scripts/compiled_value_ownership_oracle.py --phase 3 --require-hip`
through the selected managed interpreter. Success requires exit zero and the
final line `COMPILED VALUE OWNERSHIP PHASE 3: PASS`. The oracle's test
receipt checks require execution of both `hip-caller-bytes-unchanged-hardware`
and `hip-program-owned-reuse-hardware`; ignored, skipped, and zero-match
results do not count. Missing hardware fails the command.

The workflow uploads `target/ownership-hip-ci/oracle.log` and `receipt.json`,
including the requested/observed commit, platform, tool paths, timestamps,
command, exit code, and verdict. Attach the run link and artifact to the issue that
owns the hardware acceptance (chelis#1286 and chelis#1214). The complete oracle must pass on the
implementation being accepted; the wrapper's unit tests or a queued workflow cannot
close the hardware gate.
