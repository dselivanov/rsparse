#include <RcppArmadillo.h>
#include "mapped_csc.hpp"
#include "mapped_csr.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#define GRAIN_SIZE 100
#define CSC 1
#define CSR 2
#define TOL 1e-10
#define CLASSIFICATION 1
#define REGRESSION 2
#define CLIP_VALUE 100


dMappedCSR extract_mapped_csr(Rcpp::S4 input);
dMappedCSC extract_mapped_csc(Rcpp::S4 input);
arma::fmat extract_float_matrix(Rcpp::S4 x);
arma::fvec extract_float_vector(Rcpp::S4 x);
// [[Rcpp::export]]
int omp_thread_count();

bool is_master();

const std::string currentDateTime();
