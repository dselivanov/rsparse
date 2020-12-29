#include "wrmf.hpp"

template <class T>
arma::Mat<T> drop_row(const arma::Mat<T>& X_nnz, const bool drop_last) {
  if (drop_last) {  // drop last row
    return X_nnz.head_rows(X_nnz.n_rows - 1);
  } else {  // drop first row
    return X_nnz.tail_rows(X_nnz.n_rows - 1);
  }
};

template <class T>
double initialize_biases(dMappedCSC& ConfCSC,  // modified in place
                         dMappedCSC& ConfCSR,  // modified in place
                         arma::Col<T>& user_bias, arma::Col<T>& item_bias, T lambda,
                         bool dynamic_lambda, bool non_negative,
                         bool calculate_global_bias) {
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
