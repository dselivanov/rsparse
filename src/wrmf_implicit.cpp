#include "rsparse.h"
#include "wrmf_implicit.hpp"

// [[Rcpp::export]]
double als_implicit_double(const Rcpp::S4 &m_csc_r,
                  arma::mat& X,
                  arma::mat& Y,
                  const arma::mat& XtX,
                  double lambda,
                  unsigned n_threads,
                  unsigned solver,
                  unsigned cg_steps,
                  const bool with_biases,
                  bool is_x_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_implicit<double>(
      Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps,
      with_biases, is_x_bias_last_row
    );
}

// [[Rcpp::export]]
double als_implicit_float( const Rcpp::S4 &m_csc_r,
                  Rcpp::S4 &X_,
                  Rcpp::S4 & Y_,
                  Rcpp::S4 &XtX_,
                  double lambda,
                  unsigned n_threads,
                  unsigned solver,
                  unsigned cg_steps,
                  const bool with_biases,
                  bool is_x_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  // get arma matrices which share memory with R "float" matrices
  arma::fmat X = extract_float_matrix(X_);
  arma::fmat Y = extract_float_matrix(Y_);
  arma::fmat XtX = extract_float_matrix(XtX_);
  return (double)als_implicit<float>(
      Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps,
      with_biases, is_x_bias_last_row
    );
}


