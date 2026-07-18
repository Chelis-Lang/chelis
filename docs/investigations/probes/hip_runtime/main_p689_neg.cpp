#include "chelis_runtime.h"
#include <stdint.h>
#include <inttypes.h>

extern "C" void p689_neg_i64(chelis_tensor **inputs, int n_in,
                             chelis_tensor **outputs, int n_out);

int main(void) {
    int shape[1] = { 4 };
    chelis_tensor *x = chelis_alloc(1, shape, CHELIS_I64);
    int64_t in_vals[4] = { 16777217LL, 16777219LL, 9007199254740993LL, -5LL };
    for (int i = 0; i < 4; i++) ((int64_t *)x->data)[i] = in_vals[i];

    chelis_tensor *inputs[1] = { x };
    chelis_tensor *outputs[1] = { NULL };
    p689_neg_i64(inputs, 1, outputs, 1);

    printf("op=neg dtype=int64 n=4\n");
    printf("input     :");
    for (int i = 0; i < 4; i++) printf(" %" PRId64, in_vals[i]);
    printf("\n");
    printf("expected  :");
    for (int i = 0; i < 4; i++) printf(" %" PRId64, -in_vals[i]);
    printf("\n");
    printf("hip_actual:");
    for (int i = 0; i < outputs[0]->size; i++)
        printf(" %" PRId64, ((int64_t *)outputs[0]->data)[i]);
    printf("\n");
    printf("hip_hex   :");
    for (int i = 0; i < outputs[0]->size; i++)
        printf(" 0x%016" PRIx64, (uint64_t)((int64_t *)outputs[0]->data)[i]);
    printf("\n");
    printf("out_dtype=%d out_size=%d\n", outputs[0]->dtype, outputs[0]->size);

    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}
