#include "chelis_runtime.h"
#include <stdint.h>
#include <inttypes.h>

extern "C" void p689_sum_i64(chelis_tensor **inputs, int n_in,
                             chelis_tensor **outputs, int n_out);

int main(void) {
    int shape[1] = { 4 };
    chelis_tensor *x = chelis_alloc(1, shape, CHELIS_I64);
    int64_t in_vals[4] = { 1000000000000LL, 2000000000000LL, 3000000000000LL, 4000000000000LL };
    int64_t expected = 0;
    for (int i = 0; i < 4; i++) { ((int64_t *)x->data)[i] = in_vals[i]; expected += in_vals[i]; }

    chelis_tensor *inputs[1] = { x };
    chelis_tensor *outputs[1] = { NULL };
    p689_sum_i64(inputs, 1, outputs, 1);

    printf("op=sum(axis=0) dtype=int64 n=4\n");
    printf("input     :");
    for (int i = 0; i < 4; i++) printf(" %" PRId64, in_vals[i]);
    printf("\n");
    printf("expected  : %" PRId64 "\n", expected);
    printf("hip_actual: %" PRId64 "\n", ((int64_t *)outputs[0]->data)[0]);
    printf("hip_hex   : 0x%016" PRIx64 "\n", (uint64_t)((int64_t *)outputs[0]->data)[0]);
    printf("out_dtype=%d out_size=%d\n", outputs[0]->dtype, outputs[0]->size);

    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}
