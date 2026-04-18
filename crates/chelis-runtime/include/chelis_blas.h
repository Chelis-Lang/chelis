#ifndef CHELIS_BLAS_H
#define CHELIS_BLAS_H

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

#endif
