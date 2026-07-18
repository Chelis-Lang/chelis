#include "chelis_runtime.h"
#include <stdint.h>
#include <inttypes.h>

extern "C" void control_add_i64(chelis_tensor **inputs, int n_in,
                                chelis_tensor **outputs, int n_out);

int main(void) {
    int shape[1] = { 4 };
    chelis_tensor *a = chelis_alloc(1, shape, CHELIS_I64);
    chelis_tensor *b = chelis_alloc(1, shape, CHELIS_I64);
    int64_t av[4] = { 16777217LL, 9007199254740993LL, -5LL, 1000000000000LL };
    int64_t bv[4] = { 3LL,        2LL,               10LL, 2000000000000LL };
    for (int i = 0; i < 4; i++) {
        ((int64_t *)a->data)[i] = av[i];
        ((int64_t *)b->data)[i] = bv[i];
    }

    chelis_tensor *inputs[2] = { a, b };
    chelis_tensor *outputs[1] = { NULL };
    control_add_i64(inputs, 2, outputs, 1);

    printf("op=add dtype=int64 n=4 (CONTROL: int64 kernel, must be correct)\n");
    printf("expected  :");
    for (int i = 0; i < 4; i++) printf(" %" PRId64, av[i] + bv[i]);
    printf("\n");
    printf("hip_actual:");
    for (int i = 0; i < outputs[0]->size; i++)
        printf(" %" PRId64, ((int64_t *)outputs[0]->data)[i]);
    printf("\n");

    chelis_free(a);
    chelis_free(b);
    chelis_free(outputs[0]);
    return 0;
}
