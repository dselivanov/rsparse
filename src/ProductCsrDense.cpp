#include "rsparse.h"

// implements x * t(y) multiplication where x = sparse CSR matrix and y = dense matrix
// it doesn't implement just x * y because it will be slow due to the fact that
// our main pattern for access elements in y is by row which is cache inefficient and problematic to vectorize
// because matrices are stored by column
// so essentially it implements R's `tcrossprod()`
// FIXME for now works with just double but can be templated in future

// [[Rcpp::export]]
Rcpp::NumericMatrix csr_dense_tcrossprod(const Rcpp::S4 &x_csr_r, const arma::Mat<double>& y_transposed, int num_threads = 1) {
  const dMappedCSR x_csr = extract_mapped_csr(x_csr_r);
  Rcpp::NumericMatrix res(x_csr.n_rows, y_transposed.n_rows); //y_transposed.n_rows = y_dense.n_cols
  arma::dmat res_arma_map = arma::dmat(res.begin(), res.nrow(), res.ncol(), false, true);
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(num_threads)  schedule(dynamic, GRAIN_SIZE)
  #endif
  for (uint32_t i = 0; i < x_csr.n_rows; i++) {
    const uint32_t p1 = x_csr.row_ptrs[i];
    const uint32_t p2 = x_csr.row_ptrs[i + 1];
    // mapped indices are uint32_t, but arma only allows indices be uvec = vec<uword> = vec<size_t>
    // so we need to construct these indices by copying from uint32_t to uword
    const arma::Col<uint32_t> idx_temp = arma::Col<uint32_t>(&x_csr.col_indices[p1], p2 - p1);
    const arma::uvec idx = arma::conv_to<arma::uvec>::from(idx_temp);
    const arma::colvec x_csr_row = arma::colvec(&x_csr.values[p1], p2 - p1, false, false);
    res_arma_map.row(i) = (y_transposed.cols(idx) * x_csr_row).t();
  }
  return res;
}
