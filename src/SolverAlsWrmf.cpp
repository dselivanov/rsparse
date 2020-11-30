#include "rsparse.h"
#include "nnls.hpp"
#include <string.h>

#define CHOLESKY 0
#define CONJUGATE_GRADIENT 1
#define SEQ_COORDINATE_WISE_NNLS 2

#define CG_TOL 1e-10

template <class T>
arma::Mat<T> drop_row(const arma::Mat<T> &X_nnz, const bool drop_last) {
  if (drop_last) { // drop last row
    return X_nnz.head_rows(X_nnz.n_rows - 1);
  } else { // drop first row
    return X_nnz.tail_rows(X_nnz.n_rows - 1);
  }
};

template <class T>
arma::Col<T> cg_solver_impicit(const arma::Mat<T> &X_nnz,
                      const arma::Col<T> &confidence,
                      const arma::Col<T> &x_old,
                      const arma::uword n_iter,
                      const arma::Mat<T> &XtX) {
  arma::Col<T> x = x_old;
  const arma::Col<T> confidence_1 = confidence - 1.0;

  arma::Col<T> Ap;
  arma::Col<T> r = X_nnz * (confidence - (confidence_1 % (X_nnz.t() * x))) - XtX * x;
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = arma::dot(r, r);

  for(auto k = 0; k < n_iter; k++) {
    Ap = XtX * p + X_nnz * (confidence_1 % (X_nnz.t() * p));
    alpha =  rsold / dot(p, Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = dot(r, r);
    if (rsnew < CG_TOL) break;
    p = r + p * (rsnew / rsold);
    rsold = rsnew;
  }
  return x;
}

template <class T>
arma::Col<T> cg_solver_explicit(const arma::Mat<T> &X_nnz,
                                const arma::Col<T> &confidence,
                                const arma::Col<T> &x_old,
                                T lambda,
                                const arma::uword n_iter) {
  arma::Col<T> x = x_old;

  arma::Col<T> Ap;
  arma::Col<T> r = X_nnz * (confidence - (X_nnz.t() * x)) - lambda * x;
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = arma::dot(r, r);

  for(auto k = 0; k < n_iter; k++) {
    Ap = (X_nnz * (X_nnz.t() * p)) + lambda * p;
    alpha =  rsold / arma::dot(p, Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = arma::dot(r, r);
    if (rsnew < CG_TOL) break;
    p = r + p * (rsnew / rsold);
    rsold = rsnew;
  }
  return x;
}

template <class T>
T als_explicit(const dMappedCSC& Conf,
          arma::Mat<T>& X,
          arma::Mat<T>& Y,
          const double lambda,
          const unsigned n_threads,
          const unsigned solver,
          const unsigned cg_steps,
          const bool with_biases,
          const bool is_x_bias_last_row) {
  /* Note about biases:
   * For user factors, the first row will be set to all ones
   * to match with the item biases, and the calculated user biases will be in the
   * last row.
   * For item factors, the last row will be set to all ones to
   * mach with the user biases, and the calculated item biases will be in the
   * first row.
   */

  // if is_x_bias_last_row == true
  // X = [1, ..., x_bias]
  // Y = [y_bias, ..., 1]
  // if is_x_bias_last_row == false
  // X = [x_bias, ..., 1]
  // Y = [1, ..., y_bias]

  const arma::uword rank = X.n_rows;

  arma::Col<T> x_biases;

  if (with_biases) {
    if (is_x_bias_last_row) // last row
      x_biases = X.row(X.n_rows - 1).t();
    else // first row
      x_biases = X.row(0).t();
  }

  T loss = 0;
  size_t nc = Conf.n_cols;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(size_t i = 0; i < nc; i++) {
    arma::uword p1 = Conf.col_ptrs[i];
    arma::uword p2 = Conf.col_ptrs[i + 1];
    // catch situation when some columns in matrix are empty,
    // so p1 becomes equal to p2 or greater than number of columns
    if(p1 < p2) {
      auto idx = arma::uvec(&Conf.row_indices[p1], p2 - p1, false, true);
      arma::Col<T> confidence = arma::conv_to< arma::Col<T> >::from(arma::vec(&Conf.values[p1], p2 - p1));
      arma::Mat<T> X_nnz;
      // if is_x_bias_last_row == true
      // X_nnz = [1, ...]
      // if is_x_bias_last_row == false
      // X_nnz = [..., 1]
      if (with_biases) {
        X_nnz = drop_row<T>(X.cols(idx), is_x_bias_last_row);
        confidence -= x_biases(idx);
      } else {
        X_nnz = X.cols(idx);
      }

      arma::Col<T> Y_new;
      // if is_x_bias_last_row == true
      // X_nnz = [1, ..., x_bias]
      // Y_new should be [y_bias, ...]
      // if is_x_bias_last_row == false
      // X_nnz = [x_bias, ..., 1]
      // Y_new should be [..., y_bias]
      if (solver == CONJUGATE_GRADIENT) {
        if (with_biases) {
          auto init = drop_row<T>(Y.col(i), !is_x_bias_last_row);
          Y_new = cg_solver_explicit<T>(X_nnz, confidence, init, lambda, cg_steps);
        } else {
          Y_new = cg_solver_explicit<T>(X_nnz, confidence, Y.col(i), lambda, cg_steps);
        }
      } else {
        arma::Mat<T> lhs = X_nnz * X_nnz.t();
        lhs.diag() += lambda;
        const arma::Mat<T> rhs = X_nnz * confidence;

        if (solver == CHOLESKY) { // CHOLESKY
          Y_new = solve(lhs, rhs, arma::solve_opts::fast );
        } else if (solver == SEQ_COORDINATE_WISE_NNLS) { // SEQ_COORDINATE_WISE_NNLS
          Y_new = c_nnls<T>(lhs, rhs, 10000, 1e-3);
        }
      }
      arma::Row<T> err;

      if (with_biases) {
        if (is_x_bias_last_row) {
          // X_nnz = [1, ..., x_bias]
          // Y_new should be [y_bias, ...]
          // Y.col(i) should be [y_bias, ..., 1]
          Y.col(i).head(rank - 1) = Y_new;

        } else {
          // X_nnz = [x_bias, ..., 1]
          // Y_new should be [..., y_bias]
          // Y.col(i) should be [1, ..., y_bias]
          Y.col(i).tail(rank - 1) = Y_new;
        }
      } else {
        Y.col(i) = Y_new;
      }
      err = confidence.t() - (Y_new.t() * X_nnz);
      loss += arma::dot(err, err);
    } else {
      if (with_biases) {
        // FIXME should we only set to zero embeddings and leave
        // biases "as is" from initialization?
        const arma::Col<T> z(rank - 1, arma::fill::zeros);
        if (is_x_bias_last_row) {
          Y.col(i).head(rank - 1) = z;
        } else {
          Y.col(i).tail(rank - 1) = z;
        }
      } else {
        Y.col(i).zeros();
      }
    }
  }

  if(lambda > 0) {
    if (with_biases) {
      // lambda applied to all learned parameters:
      // embeddings and biases
      // so we select all rows excluding dummy ones
      // if is_x_bias_last_row == true
      // X = [1, ..., x_bias]
      // Y = [y_bias, ..., 1]
      bool is_drop_last_x = !is_x_bias_last_row;
      bool is_drop_last_y = is_x_bias_last_row;
      auto X_excl_ones = drop_row<T>(X, is_drop_last_x);
      auto Y_excl_ones = drop_row<T>(Y, is_drop_last_y);
      // accu(X_excl_ones % X_excl_ones)
      // as per arma docs "multiply-and-accumulate"
      // should should be translated
      // into efficient MKL/OpenBLAS calls
      loss += lambda * (accu(X_excl_ones % X_excl_ones) + accu(Y_excl_ones % Y_excl_ones));
    } else {
      loss += lambda * (accu(X % X) + accu(Y % Y));
    }
  }
  return (loss / Conf.nnz);
}


template <class T>
T als_impicit(const dMappedCSC& Conf,
          arma::Mat<T>& X,
          arma::Mat<T>& Y,
          const arma::Mat<T>& XtX,
          double lambda,
          unsigned n_threads,
          unsigned solver,
          unsigned cg_steps,
          bool is_x_bias_last_row) {
  // const arma::uword rank = X.n_rows;
  T loss = 0;
  size_t nc = Conf.n_cols;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(size_t i = 0; i < nc; i++) {
    arma::uword p1 = Conf.col_ptrs[i];
    arma::uword p2 = Conf.col_ptrs[i + 1];
    // catch situation when some columns in matrix are empty, so p1 becomes equal to p2 or greater than number of columns
    if(p1 < p2) {
      auto idx = arma::uvec(&Conf.row_indices[p1], p2 - p1, false, true);
      arma::Col<T> confidence = arma::conv_to< arma::Col<T> >::from(arma::vec(&Conf.values[p1], p2 - p1));
      const arma::Mat<T> X_nnz = X.cols(idx);
      arma::Col<T> Y_new;

      if(solver == CONJUGATE_GRADIENT) {
        Y_new = cg_solver_impicit<T>(X_nnz, confidence, Y.col(i), cg_steps, XtX);
      } else {
        const arma::Mat<T> lhs = XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();
        const arma::Mat<T> rhs = X_nnz * confidence;
        if (solver == SEQ_COORDINATE_WISE_NNLS) {
          Y_new = c_nnls<T>(lhs, rhs, 10000, 1e-3);
        } else { // CHOLESKY
          Y_new = solve(lhs, rhs, arma::solve_opts::fast );
        }
      }

      Y.col(i) = Y_new;
      loss += dot(square( 1 - (Y.col(i).t() * X_nnz)), confidence);

    } else {
      Y.col(i).zeros();
    }
  }

  if(lambda > 0) {
    loss += lambda * (accu(X % X) + accu(Y % Y));
  }
  return (loss / Conf.nnz);
}

// [[Rcpp::export]]
double als_implicit_double(const Rcpp::S4 &m_csc_r,
                  arma::mat& X,
                  arma::mat& Y,
                  const arma::mat& XtX,
                  double lambda,
                  unsigned n_threads,
                  unsigned solver,
                  unsigned cg_steps,
                  bool is_x_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_impicit<double>(
      Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps,
      is_x_bias_last_row
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
                  bool is_x_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  // get arma matrices which share memory with R "float" matrices
  arma::fmat X = exctract_float_matrix(X_);
  arma::fmat Y = exctract_float_matrix(Y_);
  arma::fmat XtX = exctract_float_matrix(XtX_);
  return (double)als_impicit<float>(
      Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps,
      is_x_bias_last_row
    );
}

// [[Rcpp::export]]
double als_explicit_double(const Rcpp::S4 &m_csc_r,
                           arma::mat& X,
                           arma::mat& Y,
                           double lambda,
                           unsigned n_threads,
                           unsigned solver,
                           unsigned cg_steps,
                           const bool with_biases,
                           bool is_x_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_explicit<double>(
      Conf, X, Y, lambda, n_threads, solver, cg_steps,
      with_biases, is_x_bias_last_row
  );
}

// [[Rcpp::export]]
double als_explicit_float(const Rcpp::S4 &m_csc_r,
                          Rcpp::S4 &X_,
                          Rcpp::S4 & Y_,
                          double lambda,
                          unsigned n_threads,
                          unsigned solver,
                          unsigned cg_steps,
                          const bool with_biases,
                          bool is_x_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  arma::fmat X = exctract_float_matrix(X_);
  arma::fmat Y = exctract_float_matrix(Y_);
  return (double)als_explicit<float>(
      Conf, X, Y, lambda, n_threads, solver, cg_steps,
      with_biases, is_x_bias_last_row
  );
}

template <class T>
double initialize_biases(dMappedCSC& ConfCSC, // modified in place
                         dMappedCSC& ConfCSR, // modified in place
                         arma::Col<T>& user_bias,
                         arma::Col<T>& item_bias,
                         T lambda,
                         bool non_negative,
                         bool calculate_global_bias = false) {
  /* Robust mean calculation */
  double global_bias = 0;
  if (calculate_global_bias) {
    for (size_t ix = 0; ix < ConfCSC.nnz; ix++)
      global_bias += (ConfCSC.values[ix] - global_bias) / (double)(ix+1);
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
      for (int ix = ConfCSC.col_ptrs[col]; ix < ConfCSC.col_ptrs[col+1]; ix++) {
        item_bias[col] += ConfCSC.values[ix] - user_bias[ConfCSC.row_indices[ix]];
      }
      item_bias[col] /= lambda + static_cast<T>(ConfCSC.col_ptrs[col+1] - ConfCSC.col_ptrs[col]);
      if (non_negative)
        item_bias[col] = std::fmax(0., item_bias[col]);
    }

    user_bias.zeros();
    for (int row = 0; row < ConfCSR.n_cols; row++) {
      for (int ix = ConfCSR.col_ptrs[row]; ix < ConfCSR.col_ptrs[row+1]; ix++) {
        user_bias[row] += ConfCSR.values[ix] - item_bias[ConfCSR.row_indices[ix]];
      }
      user_bias[row] /= lambda + static_cast<T>(ConfCSR.col_ptrs[row+1] - ConfCSR.col_ptrs[row]);
      if (non_negative)
        user_bias[row] = std::fmax(0., user_bias[row]);
    }
  }
  return global_bias;
}

// [[Rcpp::export]]
double initialize_biases_double(const Rcpp::S4 &m_csc_r,
                                const Rcpp::S4 &m_csr_r,
                                arma::Col<double>& user_bias,
                                arma::Col<double>& item_bias,
                                double lambda,
                                bool non_negative,
                                bool calculate_global_bias = false) {
  dMappedCSC ConfCSC = extract_mapped_csc(m_csc_r);
  dMappedCSC ConfCSR = extract_mapped_csc(m_csr_r);
  return initialize_biases<double>(ConfCSC, ConfCSR,
                                   user_bias, item_bias,
                                   lambda,
                                   non_negative,
                                   calculate_global_bias);
}

// [[Rcpp::export]]
double initialize_biases_float(const Rcpp::S4 &m_csc_r,
                               const Rcpp::S4 &m_csr_r,
                               Rcpp::S4& user_bias,
                               Rcpp::S4& item_bias,
                               double lambda,
                               bool non_negative,
                               bool calculate_global_bias = false) {
  dMappedCSC ConfCSC = extract_mapped_csc(m_csc_r);
  dMappedCSC ConfCSR = extract_mapped_csc(m_csr_r);

  arma::Col<float> user_bias_arma = exctract_float_vector(user_bias);
  arma::Col<float> item_bias_arma = exctract_float_vector(item_bias);

  return initialize_biases<float>(ConfCSC, ConfCSR,
                                  user_bias_arma,
                                  item_bias_arma,
                                  lambda,
                                  non_negative,
                                  calculate_global_bias);
}

// [[Rcpp::export]]
SEXP deep_copy(SEXP x) {
  SEXP out = PROTECT(Rf_allocVector(REALSXP, Rf_xlength(x)));
  if (Rf_xlength(x))
    memcpy(REAL(out), REAL(x), (size_t)Rf_xlength(x)*sizeof(double));
  UNPROTECT(1);
  return out;
}
