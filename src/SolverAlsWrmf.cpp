#include "rsparse.h"
#include "nnls.hpp"
#include <string.h>

template <class T>
arma::Col<T> scd_nnls_solver(const arma::Mat<T> &XtX,
                         const arma::Mat<T> &X_nnz,
                         const arma::Col<T> &confidence) {
  const arma::Mat<T> inv = XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();
  const arma::Mat<T> rhs = X_nnz * confidence;
  arma::Col<T> res;
  auto res_m = c_nnls<T>(inv, rhs, 10000, 1e-3);
  res = res_m.col(0);
  return(res);
}

template <class T>
arma::Col<T> chol_solver(const arma::Mat<T> &XtX,
                      const arma::Mat<T> &X_nnz,
                      const arma::Col<T> &confidence) {
  const arma::Mat<T> inv = XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();
  const arma::Mat<T> rhs = X_nnz * confidence;
  arma::Col<T> res = solve(inv, rhs, arma::solve_opts::fast );
  return(res);
}

template <class T>
arma::Col<T> cg_solver(const arma::Mat<T> &XtX,
                      const arma::Mat<T> &X_nnz,
                      const arma::Col<T> &confidence,
                      const arma::Col<T> &x_old,
                      const int n_iter) {
  arma::Col<T> x = x_old;
  arma::Col<T> confidence_1 = confidence - 1.0;

  arma::Mat<T> Ap;
  arma::Col<T> r = X_nnz * (confidence - (confidence_1 % (X_nnz.t() * x))) - XtX * x;
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = as_scalar(r.t() * r);

  for(int k = 0; k < n_iter; k++) {
    Ap = XtX * p + X_nnz * (confidence_1 % (X_nnz.t() * p));
    alpha =  rsold / as_scalar(p.t() * Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = as_scalar(r.t() * r);
    if (rsnew < TOL) break;
    p = r + p * (rsnew / rsold);
    rsold = rsnew;
  }
  return x;
}

template <class T>
arma::Col<T> scd_nnls_solver_explicit(const arma::Mat<T> &X_nnz,
                                      const arma::Col<T> &confidence,
                                      T regularization) {
  arma::Mat<T> inv = X_nnz * X_nnz.t();
  inv.diag() += regularization;
  const arma::Mat<T> rhs = X_nnz * confidence;
  arma::Col<T> res;
  auto res_m = c_nnls<T>(inv, rhs, 10000, 1e-3);
  res = res_m.col(0);
  return(res);
}

template <class T>
arma::Col<T> chol_solver_explicit(const arma::Mat<T> &X_nnz,
                                  const arma::Col<T> &confidence,
                                  T regularization) {
  arma::Mat<T> inv = X_nnz * X_nnz.t();
  inv.diag() += regularization;

  const arma::Mat<T> rhs = X_nnz * confidence;
  arma::Col<T> res = solve(inv, rhs, arma::solve_opts::fast );
  return(res);
}

template <class T>
arma::Col<T> cg_solver_explicit(const arma::Mat<T> &X_nnz,
                                const arma::Col<T> &confidence,
                                const arma::Col<T> &x_old,
                                T regularization,
                                const int n_iter) {
  arma::Col<T> x = x_old;

  arma::Mat<T> Ap;
  arma::Col<T> r = X_nnz * confidence - ((X_nnz * (X_nnz.t() * x)) + x * regularization);
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = as_scalar(r.t() * r);

  for(int k = 0; k < n_iter; k++) {
    Ap = (X_nnz * (X_nnz.t() * p)) + p * regularization;
    alpha =  rsold / as_scalar(p.t() * Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = as_scalar(r.t() * r);
    if (rsnew < TOL) break;
    p = r + p * (rsnew / rsold);
    rsold = rsnew;
  }
  return x;
}

template <class T>
T als_implicit_cpp(const dMappedCSC& Conf,
                    arma::Mat<T>& X,
                    arma::Mat<T>& Y,
                    const arma::Mat<T>& XtX,
                    double lambda,
                    unsigned n_threads,
                    unsigned solver,
                    unsigned cg_steps = 3,
                    bool non_negative = false) {

  if (non_negative) solver = CHOLESKY;

  if(solver != CHOLESKY && solver != CONJUGATE_GRADIENT)
    Rcpp::stop("Unknown solver code %d", solver);

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
      const arma::Col<T> confidence = arma::conv_to< arma::Col<T> >::from(arma::vec(&Conf.values[p1], p2 - p1));
      const arma::Mat<T> X_nnz = X.cols(idx);

      if (non_negative) {
        Y.col(i) = scd_nnls_solver<T>(XtX, X_nnz, confidence);
      } else {
        if(solver == CHOLESKY)
          Y.col(i) = chol_solver<T>(XtX, X_nnz, confidence);
        else if(solver == CONJUGATE_GRADIENT)
          Y.col(i) = cg_solver<T>(XtX, X_nnz, confidence, Y.col(i), cg_steps);
      }

      if(lambda >= 0)
        loss += accu(square( 1 - (Y.col(i).t() * X_nnz) ) * confidence);
    } else {
      Y.col(i).zeros();
    }
  }

  if(lambda > 0)
    loss += lambda * (accu(square(X)) + accu(square(Y)));
  return (loss / accu(arma::vec(Conf.values, Conf.nnz, false, false)));
}

template <class T>
T als_explicit_cpp(const dMappedCSC& Conf,
                   const arma::Col<double> Conf_orig,
                   arma::Mat<T>& X_,
                   arma::Mat<T>& Y_,
                   double lambda,
                   unsigned n_threads,
                   unsigned solver,
                   unsigned cg_steps = 3,
                   bool calc_user_bias = false,
                   bool calc_item_bias = false,
                   bool non_negative = false) {
  /* Note about biases: for user factors, the first column will be set to all ones
   * to match with the item biases, and the calculated user biases will be in the
   * last column. For item factors, the last column will be set to all ones to
   * mach with the user biases, and the calculated item biases will be in the
   * first column.
   */
  arma::Mat<T> X;
  arma::Mat<T> Y;
  if (calc_user_bias) {
    Y_.row(0).ones();
    for (auto col = 0; col < Conf.n_cols; col++) {
      for (auto ix = Conf.col_ptrs[col]; ix < Conf.col_ptrs[col + 1]; ix++)
        Conf.values[ix] = Conf_orig.at(ix) - X_.at(0, Conf.row_indices[ix]);
    }
    X = X_(arma::span(1, X_.n_rows-1), arma::span::all);
    Y = Y_(arma::span(1, Y_.n_rows-1), arma::span::all);
  } else if (calc_item_bias) {
    Y_.row(Y_.n_rows-1).ones();
    for (auto col = 0; col < Conf.n_cols; col++) {
      for (auto ix = Conf.col_ptrs[col]; ix < Conf.col_ptrs[col+1]; ix++)
        Conf.values[ix] = Conf_orig[ix] - X_.at(X_.n_rows-1, Conf.row_indices[ix]);
    }
    X = X_(arma::span(0, X_.n_rows-2), arma::span::all);
    Y = Y_(arma::span(0, Y_.n_rows-2), arma::span::all);
  } else {
    Y = arma::Mat<T>(Y_.memptr(), Y_.n_rows, Y_.n_cols, false, false);
    X = arma::Mat<T>(X_.memptr(), X_.n_rows, X_.n_cols, false, false);
  }

  if (non_negative) solver = CHOLESKY;

  if(solver != CHOLESKY && solver != CONJUGATE_GRADIENT)
    Rcpp::stop("Unknown solver code %d", solver);

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
      const arma::Col<T> confidence = arma::conv_to< arma::Col<T> >::from(arma::vec(&Conf.values[p1], p2 - p1));
      const arma::Mat<T> X_nnz = X.cols(idx);

      if (non_negative) {
        Y.col(i) = scd_nnls_solver_explicit<T>(X_nnz, confidence, lambda);
      } else {
        if(solver == CHOLESKY)
          Y.col(i) = chol_solver_explicit<T>(X_nnz, confidence, lambda);
        else if(solver == CONJUGATE_GRADIENT)
          Y.col(i) = cg_solver_explicit<T>(X_nnz, confidence, Y.col(i), lambda, cg_steps);
      }

      if(lambda >= 0)
        loss += accu(square( Y.col(i).t() * X_nnz - confidence.t() ));
    } else {
      Y.col(i).zeros();
    }
  }

  if (calc_user_bias) {
    X_(arma::span(1, X_.n_rows - 1), arma::span::all) = X;
    Y_(arma::span(1, Y_.n_rows - 1), arma::span::all) = Y;
    if (lambda > 0)
      loss += lambda * (
        accu(square(Y)) +
        accu(square(X(arma::span(0, X.n_rows-2), arma::span::all))));
  } else if (calc_item_bias) {
    X_(arma::span(0, X_.n_rows - 2), arma::span::all) = X;
    Y_(arma::span(0, Y_.n_rows - 2), arma::span::all) = Y;
    if (lambda > 0)
      loss += lambda * (
        accu(square(Y)) +
        accu(square(X(arma::span(1, X.n_rows-1), arma::span::all))));
  } else if (lambda > 0) {
    loss += lambda * (accu(square(X)) + accu(square(Y)));
  }

  return loss / static_cast<T>(Conf.nnz);
}

// [[Rcpp::export]]
double als_implicit_double(const Rcpp::S4 &m_csc_r,
                    arma::mat& X,
                    arma::mat& Y,
                    const arma::mat& XtX,
                    double lambda,
                    unsigned n_threads,
                    unsigned solver,
                    unsigned cg_steps = 3,
                    bool non_negative = false) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_implicit_cpp<double>(Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps, non_negative);
}

// [[Rcpp::export]]
double als_implicit_float(const Rcpp::S4 &m_csc_r,
                    Rcpp::S4 &XR,
                    Rcpp::S4 &YR,
                    Rcpp::S4 &XtXR,
                    double lambda,
                    unsigned n_threads,
                    unsigned solver,
                    unsigned cg_steps = 3,
                    bool non_negative = false) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  Rcpp::IntegerMatrix XRM = XR.slot("Data");
  Rcpp::IntegerMatrix YRM = YR.slot("Data");
  Rcpp::IntegerMatrix XtXRM = XtXR.slot("Data");
  float * x_ptr = reinterpret_cast<float *>(&XRM[0]);
  float * y_ptr = reinterpret_cast<float *>(&YRM[0]);
  float * xtx_ptr = reinterpret_cast<float *>(&XtXRM[0]);
  arma::fmat X = arma::fmat(x_ptr, XRM.nrow(), XRM.ncol(), false, true);
  arma::fmat Y = arma::fmat(y_ptr, YRM.nrow(), YRM.ncol(), false, true);
  arma::fmat XtX = arma::fmat(xtx_ptr, XtXRM.nrow(), XtXRM.ncol(), false, true);
  return (double)als_implicit_cpp<float>(Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps, non_negative);
  //#else
  //return -1.0;
  //#endif
}

// [[Rcpp::export]]
double als_explicit_double(const Rcpp::S4 &m_csc_r,
                           arma::Col<double>& m_values_orig,
                           arma::mat& X,
                           arma::mat& Y,
                           double lambda,
                           unsigned n_threads,
                           unsigned solver,
                           unsigned cg_steps = 3,
                           bool calc_user_bias = false,
                           bool calc_item_bias = false,
                           bool non_negative = false) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_explicit_cpp<double>(Conf, m_values_orig,
                                          X, Y, lambda, n_threads,
                                          solver, cg_steps,
                                          calc_user_bias, calc_item_bias,
                                          non_negative);
}

// [[Rcpp::export]]
double als_explicit_float(const Rcpp::S4 &m_csc_r,
                          arma::Col<double>& m_values_orig,
                          Rcpp::S4 &XR,
                          Rcpp::S4 & YR,
                          double lambda,
                          unsigned n_threads,
                          unsigned solver,
                          unsigned cg_steps = 3,
                          bool calc_user_bias = false,
                          bool calc_item_bias = false,
                          bool non_negative = false) {
  //#ifdef SINGLE_PRECISION_LAPACK_AVAILABLE
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  Rcpp::IntegerMatrix XRM = XR.slot("Data");
  Rcpp::IntegerMatrix YRM = YR.slot("Data");
  float * x_ptr = reinterpret_cast<float *>(&XRM[0]);
  float * y_ptr = reinterpret_cast<float *>(&YRM[0]);
  arma::fmat X = arma::fmat(x_ptr, XRM.nrow(), XRM.ncol(), false, true);
  arma::fmat Y = arma::fmat(y_ptr, YRM.nrow(), YRM.ncol(), false, true);
  return (double)als_explicit_cpp<float>(Conf, m_values_orig,
                                         X, Y, lambda, n_threads,
                                         solver, cg_steps,
                                         calc_user_bias, calc_item_bias,
                                         non_negative);
}

template <class T>
void initialize_biases(const dMappedCSC& ConfCSC,
                       const dMappedCSC& ConfCSR,
                       arma::Col<T>& user_bias,
                       arma::Col<T>& item_bias,
                       T lambda)
{
  for (int iter = 0; iter < 5; iter++) {
    item_bias.zeros();
    for (int col = 0; col < ConfCSC.n_cols; col++) {
      for (int ix = ConfCSC.col_ptrs[col]; ix < ConfCSC.col_ptrs[col+1]; ix++) {
        item_bias[col] += ConfCSC.values[ix] - user_bias[ConfCSC.row_indices[ix]];
      }
      item_bias[col] /= lambda + static_cast<T>(ConfCSC.col_ptrs[col+1] - ConfCSC.col_ptrs[col]);
    }

    user_bias.zeros();
    for (int row = 0; row < ConfCSR.n_cols; row++) {
      for (int ix = ConfCSR.col_ptrs[row]; ix < ConfCSR.col_ptrs[row+1]; ix++) {
        user_bias[row] += ConfCSR.values[ix] - item_bias[ConfCSR.row_indices[ix]];
      }
      user_bias[row] /= lambda + static_cast<T>(ConfCSR.col_ptrs[row+1] - ConfCSR.col_ptrs[row]);
    }
  }
}

// [[Rcpp::export]]
void initialize_biases_double(const Rcpp::S4 &m_csc_r,
                              const Rcpp::S4 &m_csr_r,
                              arma::Col<double>& user_bias,
                              arma::Col<double>& item_bias,
                              double lambda)
{
  const dMappedCSC ConfCSC = extract_mapped_csc(m_csc_r);
  const dMappedCSC ConfCSR = extract_mapped_csc(m_csr_r);
  initialize_biases<double>(ConfCSC, ConfCSR,
                            user_bias, item_bias,
                            lambda);
}

// [[Rcpp::export]]
void initialize_biases_float(const Rcpp::S4 &m_csc_r,
                             const Rcpp::S4 &m_csr_r,
                             Rcpp::S4& user_bias,
                             Rcpp::S4& item_bias,
                             double lambda)
{
  const dMappedCSC ConfCSC = extract_mapped_csc(m_csc_r);
  const dMappedCSC ConfCSR = extract_mapped_csc(m_csr_r);
  Rcpp::IntegerVector user_bias_ = user_bias.slot("Data");
  Rcpp::IntegerVector item_bias_ = item_bias.slot("Data");
  float * user_bias_ptr = reinterpret_cast<float *>(&user_bias_[0]);
  float * item_bias_ptr = reinterpret_cast<float *>(&item_bias_[0]);
  arma::Col<float> user_bias_arma = arma::Col<float>(user_bias_ptr, user_bias_.size(), false, true);
  arma::Col<float> item_bias_arma = arma::Col<float>(item_bias_ptr, item_bias_.size(), false, true);
  initialize_biases<float>(ConfCSC, ConfCSR,
                           user_bias_arma, item_bias_arma,
                           lambda);
}

// [[Rcpp::export]]
SEXP deep_copy(SEXP x)
{
  SEXP out = PROTECT(Rf_allocVector(REALSXP, Rf_xlength(x)));
  if (Rf_xlength(x))
    memcpy(REAL(out), REAL(x), (size_t)Rf_xlength(x)*sizeof(double));
  UNPROTECT(1);
  return out;
}
