#include "chelis_runtime.h"
#include <stdint.h>

extern "C" void sanity_neg_f32(chelis_tensor **inputs, int n_in,
                               chelis_tensor **outputs, int n_out);

int main(void) {
    int shape[1] = { 4 };
    chelis_tensor *x = chelis_alloc(1, shape, CHELIS_F32);
    float xv[4] = { 1.5f, -2.25f, 16777217.0f, 0.0f };
    for (int i = 0; i < 4; i++) x->data[i] = xv[i];

    chelis_tensor *inputs[1] = { x };
    chelis_tensor *outputs[1] = { NULL };
    sanity_neg_f32(inputs, 1, outputs, 1);

    printf("op=neg dtype=f32 n=4 (CONTROL: must be correct)\n");
    printf("input     :");
    for (int i = 0; i < 4; i++) printf(" %.6f", xv[i]);
    printf("\n");
    printf("expected  :");
    for (int i = 0; i < 4; i++) printf(" %.6f", -xv[i]);
    printf("\n");
    printf("hip_actual:");
    for (int i = 0; i < outputs[0]->size; i++) printf(" %.6f", outputs[0]->data[i]);
    printf("\n");

    chelis_free(x);
    chelis_free(outputs[0]);
    return 0;
}
