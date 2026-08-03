#include "chelis_runtime.h"
#include "chelis_blas.h"
#include "chelis_math.h"

int main(void) {
    int shape[1] = {1};
    chelis_tensor *tensor = chelis_alloc(1, shape, CHELIS_F32);
    if (tensor == NULL) {
        return 1;
    }
    chelis_free(tensor);
    return 0;
}
