# Build programs

`chelis build` checks a program, emits C, and invokes the host's C compiler
to produce an executable. For the [first program](first-program.md):

```sh
chelis build app.ch --output out/
./out/app
```

```text
Wrote out/app.c and out/app.h
Wrote out/chelis_runtime.h, out/chelis_blas.h, and out/libchelis_runtime.a
Staged runtime out/libchelis_runtime.a (sha256 0bff784080e6feed84e3843139976eeff6845a55b404b1cddcf8963a457631ef)
Compiler: /usr/bin/clang (Apple clang version 21.0.0 (clang-2100.1.1.101))
Built executable out/app
```

The executable prints what `chelis eval --file app.ch` prints. To generate
the C sources without compiling them, add `--emit-c`:

```sh
chelis build app.ch --emit-c --output out/
```

`--emit-c` writes the same sources, headers, and runtime archive, then prints
the compile command instead of running it. It needs no C compiler or archiver.
See [build requirements](install.md#build-requirements) for the
tools a normal build uses.

## Output files

For `app.ch` and `--output out/`, the build writes `out/app.c` and
`out/app.h`. A program with top-level values to print produces the executable
`out/app`. A module containing only function definitions produces the static
library `out/libapp.a` instead, with no process entry point.

The output directory also holds the runtime archive `libchelis_runtime.a`,
its public headers (`chelis_runtime.h` and the headers it includes), and
`chelis_runtime.receipt.json`, which records the archive and header hashes.
The generated header, the module archive, and the runtime come from one
compiler version; keep them together and rebuild them together.

## Numerical build flags

Every build compiles with one fixed profile: `-O2 -ffp-contract=off
-fno-fast-math` and no `-march`, so the compiler cannot fuse a multiply and
add into one rounding or reorder float arithmetic, and the instruction set does
not depend on the build machine. Transcendental functions (`exp`, `log`,
`sin`, `cos`, `tan`, `atan`, `tanh`, `erf`, `erfc`) are emitted into the
generated C as correctly rounded kernels, the same ones `chelis eval` runs, so
a built program and the evaluator return the same bits for them. Generated C
does not call the platform math library for these functions.

## Compiler selection

Set `CHELIS_CC` to choose the C compiler; otherwise Chelis uses the platform
compiler (`clang` on macOS). Static libraries use `CHELIS_AR`, or `ar`. Each
variable holds one executable name or path, with no extra arguments, and an
invalid value fails the build rather than falling back to another tool. When
the compiler is GCC, parallel loops compile with OpenMP (`-fopenmp`); this
changes speed, not results, because each parallel loop is element-wise.

## Calling a static library from C

Build a definitions-only module:

```chelis-surf
module Lib
def scale(x: f64) -> f64 = (x * 2.0f64)
```

```sh
chelis build lib.ch --output out/
```

```text
Wrote out/lib.c and out/lib.h
Wrote out/chelis_runtime.h, out/chelis_blas.h, and out/libchelis_runtime.a
Staged runtime out/libchelis_runtime.a (sha256 0bff784080e6feed84e3843139976eeff6845a55b404b1cddcf8963a457631ef)
Compiler: /usr/bin/clang (Apple clang version 21.0.0 (clang-2100.1.1.101))
Built static library out/liblib.a
Link requirements (after module archive): env SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk clang out/libchelis_runtime.a -lm
```

The `Link requirements` line lists what follows the module archive on your
link line; it prints absolute paths, shortened here. Each export's C symbol is
`chelis_fn_` followed by the lowercase hexadecimal UTF-8 bytes of its Chelis
name, so `scale` becomes `chelis_fn_7363616c65`. `out/lib.h` declares it:

```c
double chelis_fn_7363616c65(double x);
```

A C caller includes the header and links the module archive, then the
runtime, then the listed libraries:

```c
#include <stdio.h>
#include "lib.h"
int main(void) { printf("%g\n", chelis_fn_7363616c65(21.0)); return 0; }
```

```sh
clang -Iout driver.c out/liblib.a out/libchelis_runtime.a -lm -o driver
./driver
```

```text
42
```

A scalar export takes and returns C scalars, as here. Each Chelis scalar
dtype maps to one C type:

| Chelis | C |
|---|---|
| `f32`, `f64` | `float`, `double` |
| `f16`, `bf16` | `uint16_t` holding the bits; convert with `chelis_f32_to_f16` / `chelis_f16_to_f32` or `chelis_f32_to_bf16` / `chelis_bf16_to_f32` from `chelis_runtime.h` |
| `i8`, `i16`, `i32`, `i64` | `int8_t`, `int16_t`, `int32_t`, `int64_t` |
| `bool` | `bool` |

For example, `def b(x: i8, y: i16, z: i32, w: i64) -> i64` is declared as
`int64_t chelis_fn_62(int8_t x, int16_t y, int32_t z, int64_t w);`. A source-level `main` is
exported as `<name>__main`, where `<name>` is the output stem (as in
`out/<name>.h`), which leaves the name `main` free for your C
driver. With GCC, add `-fopenmp` to the caller's link line as well.

## Passing tensors and tuples

A tensor parameter or result is a `chelis_tensor *`, and a tuple result is a
`chelis_tuple *`. This module exports one of each:

```chelis-surf
module Vec
def double_all(x: tensor[3, f32]) -> tensor[3, f32] = add(x, x)
def stats(x: tensor[3, f32]) -> (f32, f32) = (tensor_to_scalar(sum(x, 0i32)), tensor_to_scalar(max_reduce(x, 0i32)))
```

`out/vec.h` declares:

```c
chelis_tensor* chelis_fn_646f75626c655f616c6c(chelis_tensor* x);
chelis_tuple* chelis_fn_7374617473(chelis_tensor* x);
```

The generated header does not include the runtime header, so include
`chelis_runtime.h` first. The caller below builds a tensor, calls both
exports, reads the results, and releases everything it owns:

```c
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
#include "vec.h"

int main(void) {
    int64_t shape[1] = {3};
    chelis_tensor *x = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *w = chelis_tensor_begin_write(x);
    float *in = chelis_tensor_write_view(w).data;
    in[0] = 1.0f; in[1] = 2.0f; in[2] = 5.0f;
    chelis_tensor_end_write(w);

    chelis_tensor *y = chelis_fn_646f75626c655f616c6c(x);
    const float *out = chelis_tensor_read_view(y).data;
    printf("double_all: %g %g %g\n", out[0], out[1], out[2]);

    chelis_tuple *t = chelis_fn_7374617473(x);
    for (int64_t i = 0; i < chelis_tuple_len(t); i++) {
        chelis_value v = chelis_tuple_get(t, i);
        float f; uint32_t b = (uint32_t)v.payload.scalar.bits; memcpy(&f, &b, 4);
        printf("stats.%lld: tag %d, value %g\n", (long long)i, v.tag, f);
    }

    chelis_tuple_release(t);
    chelis_tensor_release(y);
    chelis_tensor_release(x);
    return 0;
}
```

```text
double_all: 2 4 10
stats.0: tag 1, value 8
stats.1: tag 1, value 5
```

The rules the example follows:

- `chelis_alloc(rank, shape, dtype)` returns a new row-major tensor with a
  reference count of one. `shape` points to `rank` extents; rank 0 (with
  `shape` `NULL`) is a one-element scalar tensor, and a zero extent gives an
  empty tensor. A negative extent, or an element
  count whose byte size does not fit in `i64`, prints an error such as
  `Domain: chelis_alloc has negative extent -1 at axis 0` or
  `Overflow: chelis_alloc extent product exceeds i64` to stderr and ends the
  process with exit status 1. Other runtime failures, such as
  `chelis_tuple_get` with an index outside `0` to `chelis_tuple_len(t) - 1`
  (`tuple index out of bounds`), end the process the same way. The dtype constants are `CHELIS_DTYPE_F32`, `_F64`,
  `_BF16`, `_F16`, `_I8`, `_I16`, `_I32`, `_I64`, and `_BOOL`.
- Write elements between `chelis_tensor_begin_write` and
  `chelis_tensor_end_write`, through the `data` pointer of
  `chelis_tensor_write_view`. Read them through `chelis_tensor_read_view`,
  whose `count` field is the element count. `chelis_tensor_begin_write`
  requires that the caller hold the only reference to the tensor, so write
  before passing it to an export, as above. A write view's pointer is valid
  until `chelis_tensor_end_write`. A read view's pointer is valid until the
  tensor's last reference is released or a write begins on it; copy the data
  out if you need it longer.
- An export does not take ownership of its arguments: it retains what it
  needs, so the caller still releases `x`, and `x` can be passed again, as
  above. Every tensor or tuple an export returns belongs to the caller and is
  released with `chelis_tensor_release` or `chelis_tuple_release`.
- `chelis_tuple_get(t, i)` returns a `chelis_value`. A scalar item has tag
  `CHELIS_VALUE_SCALAR` (1) and holds its IEEE bits or integer value in
  `payload.scalar.bits`, which is how the loop reads each `f32`.

## When `check` passes and `build` fails

`chelis check` validates types, shapes, and effects; `chelis eval --file`
runs the program in the evaluator. A C build supports a narrower set of forms
than the evaluator:

- `reduce_window_sum`, `reduce_window_mean`, `reduce_window_max`, and
  `reduce_window_min` build only on `f32` tensors. On any other dtype the
  build stops with `supports reduce_window_* (...) on f32 tensors only`; cast
  to `f32` before the windowed reduction.
- Some gradients through local `List` arguments and ADT parameters do not
  build; [Transforms](transforms.md#execution-limits) lists them.

When a program uses an unsupported form, `chelis build` stops with an error
naming the operation; it never substitutes a different calculation. Run
`chelis build` early on a program you intend to ship natively.
