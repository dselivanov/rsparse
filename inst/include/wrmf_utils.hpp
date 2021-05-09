#include "wrmf.hpp"

template <class T>
arma::Mat<T> drop_row(const arma::Mat<T>& X_nnz, const bool drop_last) {
  if (drop_last) {  // drop last row
    return X_nnz.head_rows(X_nnz.n_rows - 1);
  } else {  // drop first row
    return X_nnz.tail_rows(X_nnz.n_rows - 1);
  }
};

/* https://en.wikipedia.org/wiki/Kahan_summation_algorithm */
template <class T>
long double compensated_sum(T *arr, int n)
{
  long double err = 0.;
  long double diff = 0.;
  long double temp;
  long double res = 0;

  for (int ix = 0; ix < n; ix++)
  {
    diff = arr[ix] - err;
    temp = res + diff;
    err = (temp - res) - diff;
    res = temp;
  }

  return res;
}

template <class T>
double initialize_biases_explicit(dMappedCSC& ConfCSC,  // modified in place
                                  dMappedCSC& ConfCSR,  // modified in place
                                  arma::Col<T>& user_bias, arma::Col<T>& item_bias, T lambda,
                                  bool dynamic_lambda, bool non_negative,
                                  bool calculate_global_bias)
{
  /* Robust mean calculation */
  double global_bias = 0;
  if (calculate_global_bias) {
    for (size_t ix = 0; ix < ConfCSC.nnz; ix++)
      global_bias += (ConfCSC.values[ix] - global_bias) / (double)(ix + 1);

#ifdef _OPENMP
#pragma omp simd
#endif
    for (size_t ix = 0; ix < ConfCSC.nnz; ix++) {
      ConfCSC.values[ix] -= global_bias;
      ConfCSR.values[ix] -= global_bias;
    }
  }

  for (int iter = 0; iter < 5; iter++) {
    item_bias.zeros();
    for (int col = 0; col < ConfCSC.n_cols; col++) {
      T lambda_use = lambda * (dynamic_lambda ? static_cast<T>(ConfCSC.col_ptrs[col + 1] -
                                                               ConfCSC.col_ptrs[col])
                                              : 1.);
      for (int ix = ConfCSC.col_ptrs[col]; ix < ConfCSC.col_ptrs[col + 1]; ix++) {
        item_bias[col] += ConfCSC.values[ix] - user_bias[ConfCSC.row_indices[ix]];
      }
      item_bias[col] /=
          lambda_use + static_cast<T>(ConfCSC.col_ptrs[col + 1] - ConfCSC.col_ptrs[col]);
      if (non_negative) item_bias[col] = std::fmax(0., item_bias[col]);
    }

    user_bias.zeros();
    for (int row = 0; row < ConfCSR.n_cols; row++) {
      T lambda_use = lambda * (dynamic_lambda ? static_cast<T>(ConfCSR.col_ptrs[row + 1] -
                                                               ConfCSR.col_ptrs[row])
                                              : 1.);
      for (int ix = ConfCSR.col_ptrs[row]; ix < ConfCSR.col_ptrs[row + 1]; ix++) {
        user_bias[row] += ConfCSR.values[ix] - item_bias[ConfCSR.row_indices[ix]];
      }
      user_bias[row] /=
          lambda_use + static_cast<T>(ConfCSR.col_ptrs[row + 1] - ConfCSR.col_ptrs[row]);
      if (non_negative) user_bias[row] = std::fmax(0., user_bias[row]);
    }
  }
  return global_bias;
}

template <class T>
double initialize_biases_implicit(dMappedCSC& ConfCSC, dMappedCSC& ConfCSR,
                                  arma::Col<T>& user_bias, arma::Col<T>& item_bias,
                                  T lambda, bool calculate_global_bias, bool non_negative,
                                  const bool initialize_item_biases)
{
  double global_bias = 0;
  if (calculate_global_bias) {
    long double s = compensated_sum(ConfCSR.values, ConfCSR.nnz);
    global_bias = s / (s + (long double)ConfCSR.n_rows*(long double)ConfCSR.n_cols - (long double)ConfCSR.nnz);
  }
  if (non_negative) global_bias = std::fmax(0., global_bias); /* <- should not happen, but just in case */

  user_bias.zeros();
  item_bias.zeros();

  double sweight;
  const double n_items = ConfCSR.n_rows;

  for (int row = 0; row < ConfCSR.n_cols; row++) {
    sweight = 0;
    for (int ix = ConfCSR.col_ptrs[row]; ix < ConfCSR.col_ptrs[row + 1]; ix++) {
      user_bias[row] += ConfCSR.values[ix] + global_bias * (1. - ConfCSR.values[ix]);
      sweight += ConfCSR.values[ix] - 1.;
    }
    user_bias[row] -= global_bias * n_items;
    user_bias[row] /= sweight + n_items + lambda;
    user_bias[row] /= 3; /* <- item biases are unaccounted for, don't want to assign everything to the user */
    if (non_negative) user_bias[row] = std::fmax(0., user_bias[row]);
  }

  const double n_users = ConfCSC.n_rows;
  for (int col = 0; col < ConfCSC.n_cols; col++) {
    sweight = 0;
    for (int ix = ConfCSC.col_ptrs[col]; ix < ConfCSC.col_ptrs[col + 1]; ix++) {
      item_bias[col] += ConfCSC.values[ix] + global_bias * (1. - ConfCSC.values[ix]);
      sweight += ConfCSC.values[ix] - 1.;
    }
    item_bias[col] -= global_bias * n_users;
    item_bias[col] /= sweight + n_users + lambda;
    item_bias[col] /= 3; /* <- user biases are unaccounted for */
    if (non_negative) item_bias[col] = std::fmax(0., item_bias[col]);
  }

  return global_bias;
}


template <class T>
double initialize_biases(dMappedCSC& ConfCSC,  // modified in place
                         dMappedCSC& ConfCSR,  // modified in place
                         arma::Col<T>& user_bias, arma::Col<T>& item_bias, T lambda,
                         bool dynamic_lambda, bool non_negative,
                         bool calculate_global_bias, bool is_explicit_feedback,
                         const bool initialize_item_biases) {
  if (is_explicit_feedback)
    return initialize_biases_explicit(ConfCSC, ConfCSR, user_bias, item_bias,
                                      lambda, dynamic_lambda, non_negative,
                                      calculate_global_bias);
  else
    return initialize_biases_implicit(ConfCSC, ConfCSR, user_bias, item_bias, lambda,
                                      calculate_global_bias,non_negative, initialize_item_biases);
}
