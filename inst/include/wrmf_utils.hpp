#include "wrmf.hpp"
#include <float.h>

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
      if (non_negative) item_bias[col] = std::fmax((T)0, item_bias[col]);
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
      if (non_negative) user_bias[row] = std::fmax((T)0, user_bias[row]);
    }
  }
  return global_bias;
}

template <class T>
double initialize_biases_implicit(dMappedCSC& ConfCSC, dMappedCSC& ConfCSR,
                                  arma::Col<T>& user_bias, arma::Col<T>& item_bias,
                                  T lambda, bool calculate_global_bias, bool non_negative)
{
  double global_bias = 0;
  if (calculate_global_bias) {
    long double s = compensated_sum(ConfCSR.values, ConfCSR.nnz);
    global_bias = s / (s + (long double)ConfCSR.n_rows*(long double)ConfCSR.n_cols - (long double)ConfCSR.nnz);
  }
  if (non_negative) global_bias = std::fmax(0., global_bias); /* <- should not happen, but just in case */

  const int n_users = ConfCSR.n_cols;
  const int n_items = ConfCSR.n_rows;
  std::vector<double> user_means(n_users);
  std::vector<double> item_means(n_items);
  std::vector<double> user_adjustment(n_users, DBL_EPSILON); /* <- avoid division by zero */
  std::vector<double> item_adjustment(n_items, DBL_EPSILON); /* <- avoid division by zero */
  for (int row = 0; row < n_users; row++) {
    for (int ix = ConfCSR.col_ptrs[row]; ix < ConfCSR.col_ptrs[row + 1]; ix++)
      user_adjustment[row] += ConfCSR.values[ix];
    user_adjustment[row] /= (user_adjustment[row] + (n_items - (ConfCSR.col_ptrs[row + 1] - ConfCSR.col_ptrs[row])));
    user_means[row] = user_adjustment[row];
    user_adjustment[row] *= (user_adjustment[row] / (user_adjustment[row] + lambda));
  }
  for (int col = 0; col < n_items; col++) {
    for (int ix = ConfCSC.col_ptrs[col]; ix < ConfCSC.col_ptrs[col + 1]; ix++)
      item_adjustment[col] += ConfCSC.values[ix];
    item_adjustment[col] /= (item_adjustment[col] + (n_users - (ConfCSC.col_ptrs[col + 1] - ConfCSC.col_ptrs[col])));
    item_means[col] = item_adjustment[col];
    item_adjustment[col] *= (item_adjustment[col] / (item_adjustment[col] + lambda));
  }


  double bias_mean;
  double bias_this;
  double wsum;
  for (int iter = 0; iter < 5; iter++) {
    /* item biases */
    bias_mean = 0;
    if (iter > 0) {
      for (int row = 0; row < n_users; row++)
        bias_mean += (user_bias[row] - bias_mean) / (T)(row + 1);
    }
    for (int col = 0; col < n_items; col++) {
      wsum = n_users;
      bias_this = bias_mean;
      for (int ix = ConfCSC.col_ptrs[col]; ix < ConfCSC.col_ptrs[col + 1]; ix++)
        bias_this += ((ConfCSC.values[ix] - 1) * (user_bias[ConfCSC.row_indices[ix]] - bias_this)) / (wsum += (ConfCSC.values[ix] - 1));
      item_bias[col] = (item_means[col] - bias_this - global_bias) * item_adjustment[col];
    }

    if (non_negative)
      for (int col = 0; col < n_items; col++) item_bias[col] = std::fmax((T)0, item_bias[col]);

    /* user biases */
    bias_mean = 0;
    if (iter > 0) {
      for (int col = 0; col < n_items; col++)
        bias_mean += (item_bias[col] - bias_mean) / (T)(col + 1);
    }
    for (int row = 0; row < n_users; row++) {
      wsum = n_items;
      bias_this = bias_mean;
      for (int ix = ConfCSR.col_ptrs[row]; ix < ConfCSR.col_ptrs[row + 1]; ix++)
        bias_this += ((ConfCSR.values[ix] - 1) * (item_bias[ConfCSR.row_indices[ix]] - bias_this)) / (wsum += (ConfCSR.values[ix] - 1));
      user_bias[row] = (user_means[row] - bias_this - global_bias) * user_adjustment[row];
    }

    if (non_negative)
      for (int row = 0; row < n_users; row++) user_bias[row] = std::fmax((T)0, user_bias[row]);
  }

  return global_bias;
}


template <class T>
double initialize_biases(dMappedCSC& ConfCSC,  // modified in place
                         dMappedCSC& ConfCSR,  // modified in place
                         arma::Col<T>& user_bias, arma::Col<T>& item_bias, T lambda,
                         bool dynamic_lambda, bool non_negative,
                         bool calculate_global_bias, bool is_explicit_feedback) {
  if (is_explicit_feedback)
    return initialize_biases_explicit(ConfCSC, ConfCSR, user_bias, item_bias,
                                      lambda, dynamic_lambda, non_negative,
                                      calculate_global_bias);
  else
    return initialize_biases_implicit(ConfCSC, ConfCSR, user_bias, item_bias, lambda,
                                      calculate_global_bias,non_negative);
}
