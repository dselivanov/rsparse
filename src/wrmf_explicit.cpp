#include "wrmf_explicit.hpp"
#include "rsparse.h"

// [[Rcpp::export]]
double als_explicit_double(const Rcpp::S4& m_csc_r, arma::mat& X, arma::mat& Y,
                           arma::Col<double> cnt_X, double lambda, double lambda_l1, unsigned int n_threads,
                           unsigned int solver, unsigned int cg_steps, unsigned int cd_steps,
                           const bool cd_until_conv, const bool dynamic_lambda,
                           const bool with_biases, bool is_x_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_explicit<double>(Conf, X, Y, lambda, lambda_l1, n_threads, solver,
                                      cg_steps, cd_steps, dynamic_lambda, cnt_X, with_biases,
                                      is_x_bias_last_row, cd_until_conv);
}

// [[Rcpp::export]]
double als_explicit_float(const Rcpp::S4& m_csc_r, Rcpp::S4& X_, Rcpp::S4& Y_,
                          Rcpp::S4& cnt_X_, double lambda, double lambda_l1, unsigned n_threads,
                          unsigned int solver, unsigned int cg_steps, unsigned int cd_steps,
                          const bool cd_until_conv, const bool dynamic_lambda,
                          const bool with_biases, bool is_x_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  arma::fmat X = extract_float_matrix(X_);
  arma::fmat Y = extract_float_matrix(Y_);
  arma::fmat cnt_X = extract_float_vector(cnt_X_);
  return (double)als_explicit<float>(Conf, X, Y, lambda, lambda_l1, n_threads, solver,
                                     cg_steps, cd_steps, dynamic_lambda, cnt_X, with_biases,
                                     is_x_bias_last_row, cd_until_conv);
}
