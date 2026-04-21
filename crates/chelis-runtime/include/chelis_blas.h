#ifndef CHELIS_BLAS_H
#define CHELIS_BLAS_H

#ifdef __APPLE__
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#elif defined(__has_include)
#if __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#else
#error "cblas.h not found; install openblas-devel or libopenblas-dev"
#endif
#else
#include <cblas.h>
#endif

#endif
