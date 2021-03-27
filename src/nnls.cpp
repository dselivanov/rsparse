#include "nnls.hpp"
#include <RcppArmadillo.h>

// [[Rcpp::export]]
arma::Mat<double> c_nnls_double(const arma::mat& x, const arma::vec& y, unsigned int max_iter,
                                double rel_tol) {
  auto n = y.size();
  Rcpp::NumericVector res(n);
  for (auto i = 0; i < n; i++) {
    res[i] = R::runif(0.0, 0.01);
  }

  arma::colvec init(&res[0], n, false, true);
  return c_nnls<double>(x, y, init, max_iter, rel_tol);
}
