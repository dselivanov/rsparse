#define GRAIN_SIZE 100
#include "MappedCSR.h"
using namespace Rcpp;


// implements x_csr * y_dense multiplication
// inside our main patter for access to y_dense is by row
// (slow due to cache inefficiency and lack of vectorization because matrices in R stored by columns)
// so we transpose y_dense on R code in order to make access by column
// so essentially it implements `tcrossprod()` - see ?tcrossprod
// FIXME for now works with just double but can be templated in future

// [[Rcpp::export]]
NumericMatrix csr_dense_tcrossprod(const S4 &x_csr_r, const arma::Mat<double>& y_transposed, int num_threads = 1) {
  const dMappedCSR x_csr = extract_mapped_csr(x_csr_r);
  NumericMatrix res(x_csr.n_rows, y_transposed.n_rows); //y_transposed.n_rows = y_dense.n_cols
  arma::dmat res_arma_map = arma::dmat(res.begin(), res.nrow(), res.ncol(), false, true);
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(num_threads)  schedule(dynamic, GRAIN_SIZE)
  #endif
  for (uint32_t i = 0; i < x_csr.n_rows; i++) {
    const uint32_t p1 = x_csr.p[i];
    const uint32_t p2 = x_csr.p[i + 1];
    // mapped indices are uint32_t, but arma only allows indices be uvec = vec<uword> = vec<size_t>
    // so we need to construct these indices by copying from uint32_t to uword
    const arma::Col<uint32_t> idx_temp = arma::Col<uint32_t>(&x_csr.j[p1], p2 - p1);
    const arma::uvec idx = arma::conv_to<arma::uvec>::from(idx_temp);
    const arma::colvec x_csr_row = arma::colvec(&x_csr.x[p1], p2 - p1, false, false);
    res_arma_map.row(i) = (y_transposed.cols(idx) * x_csr_row).t();
  }
  return res;
}
