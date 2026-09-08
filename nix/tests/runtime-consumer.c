#include "chelis_runtime.h"
#include "chelis_blas.h"
#include "chelis_math.h"

#include <stdint.h>

int main(void) {
    int64_t shape[1] = {1};
    chelis_tensor *tensor = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    if (tensor == NULL) {
        return 1;
    }
    chelis_tensor_release(tensor);
    return 0;
}
