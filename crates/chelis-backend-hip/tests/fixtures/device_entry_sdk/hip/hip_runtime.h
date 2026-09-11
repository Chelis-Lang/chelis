/* CPU execution fixture only; kernel text is compiled by the test's C++ compiler. */
#ifndef CHELIS_ENTRY_FIXTURE_HIP_H
#define CHELIS_ENTRY_FIXTURE_HIP_H
#include "hip_runtime_api.h"
struct dim3 {
    unsigned int x, y, z;
    explicit dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1)
        : x(x_), y(y_), z(z_) {}
};
#endif
