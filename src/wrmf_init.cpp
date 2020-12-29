#include "rsparse.h"
#include "wrmf.hpp"
#include "wrmf_utils.hpp"

// [[Rcpp::export]]
double initialize_biases_double(const Rcpp::S4& m_csc_r, const Rcpp::S4& m_csr_r,
                                arma::Col<double>& user_bias,
                                arma::Col<double>& item_bias, double lambda,
                                bool dynamic_lambda, bool non_negative,
                                bool calculate_global_bias = false) {
  dMappedCSC ConfCSC = extract_mapped_csc(m_csc_r);
  dMappedCSC ConfCSR = extract_mapped_csc(m_csr_r);
  return initialize_biases<double>(ConfCSC, ConfCSR, user_bias, item_bias, lambda,
                                   dynamic_lambda, non_negative, calculate_global_bias);
}

// [[Rcpp::export]]
double initialize_biases_float(const Rcpp::S4& m_csc_r, const Rcpp::S4& m_csr_r,
                               Rcpp::S4& user_bias, Rcpp::S4& item_bias, double lambda,
                               bool dynamic_lambda, bool non_negative,
                               bool calculate_global_bias = false) {
  dMappedCSC ConfCSC = extract_mapped_csc(m_csc_r);
  dMappedCSC ConfCSR = extract_mapped_csc(m_csr_r);

  arma::Col<float> user_bias_arma = extract_float_vector(user_bias);
  arma::Col<float> item_bias_arma = extract_float_vector(item_bias);

  return initialize_biases<float>(ConfCSC, ConfCSR, user_bias_arma, item_bias_arma,
                                  lambda, dynamic_lambda, non_negative,
                                  calculate_global_bias);
}
