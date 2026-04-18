#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /path/to/chelis" >&2
  exit 2
fi

chelis_bin="$1"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

cat > "$tmpdir/matmul.ch" <<'EOF'
a = (a : tensor[2, 3, f32])
b = (b : tensor[3, 4, f32])
out = (matmul(a, b) : tensor[2, 4, f32])
EOF

"$chelis_bin" build "$tmpdir/matmul.ch" --output "$tmpdir/out"
grep -q 'cblas_sgemm' "$tmpdir/out/matmul.c"
grep -q 'chelis_blas.h' "$tmpdir/out/matmul.c"

cat > "$tmpdir/out/driver.c" <<'EOF'
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
        fprintf(stderr, "matmul did not populate the result output\n");
        return 1;
    }
    for (int i = 0; i < 8; ++i) {
        if (fabsf(outputs[2]->data[i] - expected[i]) > 1e-4f) {
            fprintf(stderr, "matmul mismatch at %d: got %f expected %f\n", i, outputs[2]->data[i], expected[i]);
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
EOF

clang -O2 -I "$tmpdir/out" -c "$tmpdir/out/matmul.c" -o "$tmpdir/out/matmul.o"
clang -O2 -I "$tmpdir/out" -c "$tmpdir/out/driver.c" -o "$tmpdir/out/driver.o"
clang "$tmpdir/out/matmul.o" "$tmpdir/out/driver.o" \
  "$tmpdir/out/libchelis_runtime.a" -framework Accelerate \
  -o "$tmpdir/out/matmul_bin"
"$tmpdir/out/matmul_bin"
