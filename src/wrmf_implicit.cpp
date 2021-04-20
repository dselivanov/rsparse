#include "wrmf_implicit.hpp"
#include "rsparse.h"

// [[Rcpp::export]]
double als_implicit_double(const Rcpp::S4& m_csc_r, arma::mat& X, arma::mat& Y,
                           const arma::mat& XtX, double lambda, int n_threads,
                           const unsigned int solver, const unsigned int cg_steps, const bool with_biases,
                           const bool is_x_bias_last_row, const double global_bias,
                           arma::vec& global_bias_base, const bool initialize_bias_base) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_implicit<double>(Conf, X, Y, XtX, lambda, n_threads, solver,
                                      cg_steps, with_biases, is_x_bias_last_row,
                                      global_bias, global_bias_base, initialize_bias_base);
}

// [[Rcpp::export]]
double als_implicit_float(const Rcpp::S4& m_csc_r, Rcpp::S4& X_, Rcpp::S4& Y_,
                          Rcpp::S4& XtX_, double lambda, int n_threads,
                          const unsigned int solver, const unsigned int cg_steps, const bool with_biases,
                          const bool is_x_bias_last_row, const double global_bias,
                          Rcpp::S4& global_bias_base_, const bool initialize_bias_base) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  // get arma matrices which share memory with R "float" matrices
  arma::fmat X = extract_float_matrix(X_);
  arma::fmat Y = extract_float_matrix(Y_);
  arma::fmat XtX = extract_float_matrix(XtX_);
  arma::fvec global_bias_base = extract_float_vector(global_bias_base_);
  return (double)als_implicit<float>(Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps,
                                     with_biases, is_x_bias_last_row,
                                     global_bias, global_bias_base, initialize_bias_base);
}
