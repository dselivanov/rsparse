#include <RcppArmadillo.h>
#include <queue>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define GRAIN_SIZE 10

#define CSC 1
#define CSR 2

using namespace Rcpp;
using namespace RcppArmadillo;
using namespace arma;

// for sparse_hash_map < <uint32_t, uint32_t>, T >
#include <unordered_map>
#include <utility>

// [[Rcpp::export]]
NumericVector cpp_make_sparse_approximation(const S4 &mat_template,
                                            arma::mat& X,
                                            arma::mat& Y,
                                            int sparse_matrix_type,
                                            unsigned n_threads) {
  IntegerVector rp = mat_template.slot("p");
  int* p = rp.begin();
  IntegerVector rj;
  if(sparse_matrix_type == CSR) {
    rj = mat_template.slot("j");
  } else if(sparse_matrix_type == CSC) {
    rj = mat_template.slot("i");
  } else
    ::Rf_error("make_sparse_approximation_csr doesn't know sparse matrix type. Should be CSC=1 or CSR=2");

  uint32_t* j = (uint32_t *)rj.begin();
  IntegerVector dim = mat_template.slot("Dim");

  size_t nr = dim[0];
  size_t nc = dim[1];
  uint32_t N;
  if(sparse_matrix_type == CSR)
    N = nr;
  else
    N = nc;

  NumericVector approximated_values(rj.length());

  double *ptr_approximated_values = approximated_values.begin();
  double loss = 0;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(uint32_t i = 0; i < N; i++) {
    int p1 = p[i];
    int p2 = p[i + 1];
    for(int pp = p1; pp < p2; pp++) {
      uint64_t ind = (size_t)j[pp];
      if(sparse_matrix_type == CSR)
        ptr_approximated_values[pp] = as_scalar(X.col(i).t() * Y.col(ind));
      else
        ptr_approximated_values[pp] = as_scalar(Y.col(i).t() * X.col(ind));
    }

  }
  return(approximated_values);
}

// [[Rcpp::export]]
List  arma_svd_econ(const arma::mat& X) {
  int k = std::min(X.n_rows, X.n_cols);
  NumericMatrix UR(X.n_rows, k);
  NumericMatrix VR(X.n_cols, k);
  NumericVector dR(k);
  arma::mat U(UR.begin(), UR.nrow(), UR.ncol(),  false, true);
  arma::mat V(VR.begin(), VR.nrow(), VR.ncol(),  false, true);
  arma::vec d(dR.begin(), dR.size(),  false, true);
  int status = svd_econ(U, d, V, X);
  if(!status)
    ::Rf_error("arma::svd_econ failed");
  return(List::create(_["d"] = dR, _["u"] = UR, _["v"] = VR));
}
