#include "nnls.hpp"
#include <RcppArmadillo.h>

// [[Rcpp::export(rng=false)]]
arma::Mat<double> c_nnls_double(const arma::mat &x,
                                const arma::mat &y,
                                uint max_iter,
                                double rel_tol) {
  return c_nnls<double>(x, y, max_iter, rel_tol);
}
