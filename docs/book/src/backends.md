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

A scalar export takes and returns C scalars, as here. A tensor export uses
the `chelis_tensor` type from `chelis_runtime.h`, and a tuple result uses the
`chelis_tuple` helpers, so read each declaration in the generated header
before writing the call. A source-level `main` is exported as
`<module>__main`, which leaves the name `main` free for your C driver. With
GCC, add `-fopenmp` to the caller's link line as well.

## When `check` passes and `build` fails

`chelis check` validates types, shapes, and effects; `chelis eval --file`
runs the program in the evaluator. The C emitter supports a narrower set of
forms than the evaluator, for example some `reduce_window_*` shapes and
gradients through local `List` arguments
(see [Transforms](transforms.md#execution-limits)). When a program
uses one of these, `chelis build` stops with an error naming the operation;
it never substitutes a different calculation.
Run `chelis build` early on a program you intend to ship natively.
