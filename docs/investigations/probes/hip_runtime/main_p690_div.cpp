#include "chelis_runtime.h"
#include <stdint.h>
#include <inttypes.h>

extern "C" void p690_div_i64(chelis_tensor **inputs, int n_in,
                             chelis_tensor **outputs, int n_out);

int main(void) {
    int shape[1] = { 4 };
    chelis_tensor *a = chelis_alloc(1, shape, CHELIS_I64);
    chelis_tensor *b = chelis_alloc(1, shape, CHELIS_I64);
    int64_t av[4] = { 100LL, 42LL, -7LL, 9LL };
    int64_t bv[4] = { 0LL,   0LL,  0LL, 3LL };   // three divisors are zero
    for (int i = 0; i < 4; i++) {
        ((int64_t *)a->data)[i] = av[i];
        ((int64_t *)b->data)[i] = bv[i];
    }

    chelis_tensor *inputs[2] = { a, b };
    chelis_tensor *outputs[1] = { NULL };
    printf("op=trunc_div dtype=int64 n=4 (b[0..2]=0 -> divide by zero)\n");
    printf("dividend  :");
    for (int i = 0; i < 4; i++) printf(" %" PRId64, av[i]);
    printf("\n");
    printf("divisor   :");
    for (int i = 0; i < 4; i++) printf(" %" PRId64, bv[i]);
    printf("\n");
    fflush(stdout);
    p690_div_i64(inputs, 2, outputs, 1);   // eval path aborts here; HIP does not
    printf("hip_returned_without_abort=yes\n");
    printf("hip_actual:");
    for (int i = 0; i < outputs[0]->size; i++)
        printf(" %" PRId64, ((int64_t *)outputs[0]->data)[i]);
    printf("\n");
    printf("hip_hex   :");
    for (int i = 0; i < outputs[0]->size; i++)
        printf(" 0x%016" PRIx64, (uint64_t)((int64_t *)outputs[0]->data)[i]);
    printf("\n");

    chelis_free(a);
    chelis_free(b);
    chelis_free(outputs[0]);
    return 0;
}
