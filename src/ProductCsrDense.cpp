#include "MappedCSR.h"
using namespace Rcpp;
using namespace Eigen;

// dMappedCSR extract_mapped_csr(S4 input){
//   IntegerVector dim = input.slot("Dim");
//   NumericVector rx = input.slot("x");
//   uint32_t nrows = dim[0];
//   uint32_t ncols = dim[1];
//   IntegerVector rj = input.slot("j");
//   IntegerVector rp = input.slot("p");
//   return dMappedCSR(nrows, ncols, rx.length(), (uint32_t *)rj.begin(), (uint32_t *)rp.begin(), rx.begin());
// }

typedef Map< Eigen::SparseMatrix<double, Eigen::RowMajor> > MSpMat_csr;
MSpMat_csr csr_map(const S4 &csr) {
  dMappedCSR m = extract_mapped_csr(csr);
  // not sure why casts to int*/double* are needed
  MSpMat_csr sparse(m.n_rows, m.n_cols, m.nnz, (int *)m.p, (int *)m.j, (double *)m.x);
  return(sparse);
}

// [[Rcpp::export]]
NumericMatrix prod_csr_dense(const S4 &csr_r, const SEXP &dense_m_r) {
  const MSpMat_csr sparse = csr_map(csr_r);
  const Map<MatrixXd> dense(as<Map<MatrixXd> >(dense_m_r));
  NumericMatrix res(sparse.rows(), dense.cols());
  Map<MatrixXd>( res.begin(), res.nrow(), res.ncol()) = sparse * dense;
  return res;
}

// [[Rcpp::export]]
NumericMatrix prod_dense_csr(const SEXP &dense_m_r, const S4 &csr_r) {
  const MSpMat_csr sparse = csr_map(csr_r);
  const Map<MatrixXd> dense(as<Map<MatrixXd> >(dense_m_r));
  NumericMatrix res(dense.rows(), sparse.cols());
  Map<MatrixXd>( res.begin(), res.nrow(), res.ncol()) = dense * sparse;
  return res;
}

// [[Rcpp::export]]
NumericMatrix tcrossprod_dense_csr(const SEXP &dense_m_r, const S4 &csr_r) {
  const MSpMat_csr sparse = csr_map(csr_r);
  const Map<MatrixXd> dense(as<Map<MatrixXd> >(dense_m_r));
  NumericMatrix res(dense.rows(), sparse.rows());
  Map<MatrixXd>( res.begin(), res.nrow(), res.ncol()) = dense * sparse.transpose();
  return res;
}

// [[Rcpp::export]]
NumericMatrix crossprod_csr_dense(const S4 &csr_r, const SEXP &dense_m_r) {
  const MSpMat_csr sparse = csr_map(csr_r);
  const Map<MatrixXd> dense(as<Map<MatrixXd> >(dense_m_r));
  NumericMatrix res(sparse.cols(), dense.cols());
  // not sure why, but having sparse transposed materialized is much faster
  // than product with transposed not-materialized
  Eigen::SparseMatrix<double, Eigen::RowMajor> sparse2 = sparse.transpose();
  Map<MatrixXd>( res.begin(), res.nrow(), res.ncol()) = sparse2 * dense;
  return res;
}
