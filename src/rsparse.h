#include <RcppArmadillo.h>
#include "MappedSparseMatrices.h"
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#define GRAIN_SIZE 100
#define CSC 1
#define CSR 2
#define CHOLESKY 0
#define CONJUGATE_GRADIENT 1
#define TOL 1e-10
#define CLASSIFICATION 1
#define REGRESSION 2
#define CLIP_VALUE 100


dMappedCSR extract_mapped_csr(Rcpp::S4 input);
dMappedCSC extract_mapped_csc(Rcpp::S4 input);

// [[Rcpp::export]]
int omp_thread_count();
