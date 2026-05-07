# Local HIP Environment

This repository is currently being worked on from a real AMD/ROCm machine, not a
CPU-only dev box. HIP manual gates are locally runnable when the environment below is
intact.

## Hardware and Tools

- Host OS: Fedora 43 (`Linux fedora 6.18.16-200.fc43.x86_64`)
- CPU marketing name from `rocminfo`: `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`
- GPU marketing name from `rocminfo`: `Radeon 8060S Graphics`
- ROCm ISA from `rocminfo`: `amdgcn-amd-amdhsa--gfx1100`
- `hipcc` on PATH: HIP `7.13.26162-1140233ffe`, installed via the
  `_rocm_sdk_core` Python wheel at
  `~/.local/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin/clang++`
- `rocminfo` is available and should be treated as the source of truth for local GPU
  probing.
- `rocm-smi` may be absent; do not assume it exists before using it in instructions or
  validation scripts.

## Reconciled Toolchain

Toolchain reconciled 2026-05-01: wheel ROCm is authoritative.

The `_rocm_sdk_core` Python wheel (HIP 7.13) is the single authoritative HIP stack on
this workstation. The wheel ships its own headers, clang-23, device libraries, and
`rocminfo`/`rocm-smi`/`hipconfig` binaries; all of those land under
`~/.local/lib/python3.12/site-packages/_rocm_sdk_core/`.

The Fedora `rocm-hip-devel` package (HIP 6.4) is also installed and ships conflicting
headers under `/usr/include/hip/`. Without intervention, the wheel's clang resolves
`<hip/hip_runtime.h>` against those older system headers via clang's standard
`/usr/include` search path. The wheel's own HIP include is added with `-idirafter`,
lowest priority, and the HIP 6.4 headers reference an `__AMDGCN_WAVEFRONT_SIZE` macro
that the wheel's clang-23 no longer defines:

```text
/usr/include/hip/amd_detail/amd_warp_functions.h:96:37: error: use of
undeclared identifier '__AMDGCN_WAVEFRONT_SIZE'
```

## Reconciliation

1. Wheel-include override, durable via systemd-user `environment.d`.

   The canonical home of the HIP include override is
   `~/.config/environment.d/hip.conf`:

   ```text
   HIPCC_COMPILE_FLAGS_APPEND=-isystem /home/jeff/.local/lib/python3.12/site-packages/_rocm_sdk_core/include
   ```

   systemd's user manager reads `environment.d/*.conf` at user-session start and
   propagates the variable to every process the user manager spawns, including login
   shells, graphical sessions, non-interactive `bash -c` invocations,
   `cargo test --workspace`, and orchestrated or CI-style harnesses. `hipcc` honors that
   variable and prepends the wheel HIP include as a high-priority `-isystem` path, so
   `<hip/...>` resolves inside the wheel and the HIP 6.4 system headers are ignored.

   A previous iteration of this fix only set the variable in `~/.bashrc`, which bash
   sources only for interactive shells. Non-interactive contexts never saw it and the
   HIP test went red. The `environment.d` route covers both. The `~/.bashrc` line may be
   retained for interactive shells that bypass systemd, or removed; if retained it must
   not conflict with the `environment.d` value.

2. Wheel `.so` symlinks added.

   The wheel ships only versioned `libamdhip64.so.7`, `libhiprtc.so.7`, and
   `libhiprtc-builtins.so.7`, with no unversioned `.so` symlinks. `hipcc -lamdhip64
   -lhiprtc`, used by chelis HIP smoke tests, therefore fails at link time. Symlinks
   were added under
   `~/.local/lib/python3.12/site-packages/_rocm_sdk_core/lib/` so linking finds them. If
   the wheel is upgraded these symlinks must be recreated.

Removing the system `rocm-hip-devel` package would also work but requires sudo; the
env-var route was chosen because it is purely user-space, survives system updates, and
only affects `hipcc` invocations.

## Verification

Verify the durable fix from a fresh non-interactive shell after login:

```sh
bash -c 'env | grep HIPCC_COMPILE_FLAGS_APPEND'
bash -c 'cd /home/jeff/Documents/scratch/chelis && cargo test -p chelis-backend-hip --test codegen_structure s14_generated_hip_source_compiles_when_hipcc_available'
```

Expected:

```text
HIPCC_COMPILE_FLAGS_APPEND=-isystem /home/jeff/.local/lib/python3.12/site-packages/_rocm_sdk_core/include
... test s14_generated_hip_source_compiles_when_hipcc_available ... ok
```

Also verify resolved HIP includes:

```sh
hipcc -E foo.hip | grep amd_warp_functions.h
```

The resolved path must be the wheel one.

Acceptance check from 2026-05-01:

```sh
cat /tmp/trivial_kernel.hip
#include <hip/hip_runtime.h>
__global__ void noop() {}
hipcc -c /tmp/trivial_kernel.hip -o /tmp/trivial_kernel.o
echo $?
```

Expected exit status: `0`.

`s14_generated_hip_source_compiles_when_hipcc_available`, in
`chelis-backend-hip/tests/codegen_structure.rs`, passes from non-interactive bash once
`~/.config/environment.d/hip.conf` is present and the user session has been
re-established by login or `systemctl --user import-environment`.

## Runtime Header Caveat

Any HIP source that does not `#include <hip/hip_runtime.h>` fails because clang-23's
auto-included `__clang_hip_runtime_wrapper.h` no longer pulls in the launch helpers.
This is upstream HIP behavior, not a workstation defect. Every kernel must include the
runtime header explicitly.
