#include <armadillo>
#include "mapped_csc.hpp"
#include "mapped_csr.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#define GRAIN_SIZE 100

#define CHOLESKY 0
#define CONJUGATE_GRADIENT 1
#define SEQ_COORDINATE_WISE_NNLS 2

#define SCD_MAX_ITER 10000
#define SCD_TOL 1e-3
#define CG_TOL 1e-10
