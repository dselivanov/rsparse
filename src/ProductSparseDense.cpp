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
  arma::dmat res_arma_map = arma::dmat(res.begin(), res.nrow(), res.ncol(), false, false);
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(num_threads)  schedule(dynamic, GRAIN_SIZE)
  #endif
  for (arma::uword i = 0; i < x_csr.n_rows; i++) {
    const arma::uword p1 = x_csr.row_ptrs[i];
    const arma::uword p2 = x_csr.row_ptrs[i + 1];
    const arma::uvec idx = arma::uvec(&x_csr.col_indices[p1], p2 - p1);
    const arma::colvec x_csr_row = arma::colvec(&x_csr.values[p1], p2 - p1, false, false);
    res_arma_map.row(i) = (y_transposed.cols(idx) * x_csr_row).t();
  }
  return res;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix dense_csc_prod(const Rcpp::NumericMatrix &x_r, const Rcpp::S4 &y_csc_r, int num_threads = 1) {
  const arma::dmat x = arma::dmat((double *)&x_r[0], x_r.nrow(), x_r.ncol(), false, false);
  const dMappedCSC y_csc = extract_mapped_csc(y_csc_r);
  Rcpp::NumericMatrix res(x.n_rows, y_csc.n_cols);
  arma::dmat res_arma_map = arma::dmat(res.begin(), res.nrow(), res.ncol(), false, false);
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(num_threads)  schedule(dynamic, GRAIN_SIZE)
  #endif
  for (arma::uword i = 0; i < y_csc.n_cols; i++) {
    const arma::uword p1 = y_csc.col_ptrs[i];
    const arma::uword p2 = y_csc.col_ptrs[i + 1];
    const arma::uvec idx = arma::uvec(&y_csc.row_indices[p1], p2 - p1);
    const arma::colvec y_csc_col = arma::colvec(&y_csc.values[p1], p2 - p1, false, false);
    res_arma_map.col(i) = x.cols(idx) * y_csc_col;
  }
  return res;
}
