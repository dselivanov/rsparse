#include "rsparse.h"
#include "nnls.hpp"

#define CHOLESKY 0
#define CONJUGATE_GRADIENT 1
#define SEQ_COORDINATE_WISE_NNLS 2

template <class T>
arma::Mat<T> without_row(const arma::Mat<T> &X_nnz, const bool last) {
  if (last) {
    return X_nnz.head_rows(X_nnz.n_rows - 1);
  } else {
    return X_nnz.tail_rows(X_nnz.n_rows - 1);
  }
};

template <class T>
arma::Col<T> cg_solver_impicit(const arma::Mat<T> &X_nnz,
                      const arma::Col<T> &confidence,
                      const arma::Col<T> &x_old,
                      const int n_iter,
                      const arma::Mat<T> &XtX) {
  arma::Col<T> x = x_old;
  arma::Col<T> confidence_1 = confidence - 1.0;

  arma::Col<T> Ap;
  arma::Col<T> r = X_nnz * (confidence - (confidence_1 % (X_nnz.t() * x))) - XtX * x;
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = arma::dot(r, r);

  for(int k = 0; k < n_iter; k++) {
    Ap = XtX * p + X_nnz * (confidence_1 % (X_nnz.t() * p));
    alpha =  rsold / dot(p, Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = dot(r, r);
    if (rsnew < TOL) break;
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
          const bool is_bias_last_row) {

  const arma::uword rank = X.n_rows;

  arma::Col<T> biases;

  if (with_biases) {
    // row number where biases are stored
    arma::uword bias_index = 0;
    if (is_bias_last_row) bias_index = X.n_rows - 1;
     biases = X.row(bias_index).t();
  }

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
      arma::Mat<T> X_nnz;
      // if is_bias_last_row == true
      // X_nnz = [1, ...]
      // if is_bias_last_row == false
      // X_nnz = [..., 1]
      if (with_biases) {
        X_nnz = without_row<T>(X.cols(idx), is_bias_last_row);
        confidence -= biases(idx);
      } else {
        X_nnz = X.cols(idx);
      }

      arma::Mat<T> lhs = X_nnz * X_nnz.t();
      lhs.diag() += lambda;
      const arma::Mat<T> rhs = X_nnz * confidence;
      arma::Col<T> Y_new;
      // if is_bias_last_row == true
      // X_nnz = [1, ..., x_bias]
      // Y_new should be [y_bias, ...]
      // if is_bias_last_row == false
      // X_nnz = [x_bias, ..., 1]
      // Y_new should be [..., y_bias]
      if (solver == CHOLESKY) { // CHOLESKY
        Y_new = solve(lhs, rhs, arma::solve_opts::fast );
      } else if (solver == SEQ_COORDINATE_WISE_NNLS) { // SEQ_COORDINATE_WISE_NNLS
        Y_new = c_nnls<T>(lhs, rhs, 10000, 1e-3);
      }

      arma::Row<T> err;

      if (with_biases) {
        if (is_bias_last_row) {
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
      Y.col(i).zeros();
    }
  }

  if(lambda > 0) {
    if (with_biases) {
      auto X_no_bias = X(arma::span(1, X.n_rows - 1), arma::span::all);
      auto Y_no_bias = X(arma::span(1, Y.n_rows - 1), arma::span::all);
      loss += lambda * (accu(square(X_no_bias)) + accu(square(Y_no_bias)));
    } else {
      loss += lambda * (accu(square(X)) + accu(square(Y)));
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
          bool is_bias_last_row) {
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

      if(solver == CHOLESKY || solver == SEQ_COORDINATE_WISE_NNLS) {
        const arma::Mat<T> lhs = XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();
        const arma::Mat<T> rhs = X_nnz * confidence;
        if (solver == SEQ_COORDINATE_WISE_NNLS) { // SEQ_COORDINATE_WISE_NNLS
          Y_new = c_nnls<T>(lhs, rhs, 10000, 1e-3);
        } else { // CHOLESKY
          Y_new = solve(lhs, rhs, arma::solve_opts::fast );
        }
      } else { // CONJUGATE_GRADIENT
        Y_new = cg_solver_impicit<T>(X_nnz, confidence, Y.col(i), cg_steps, XtX);
      }
      Y.col(i) = Y_new;
      loss += dot(square( 1 - (Y.col(i).t() * X_nnz)), confidence);

    } else {
      Y.col(i).zeros();
    }
  }

  if(lambda > 0) {
    loss += lambda * (accu(square(X)) + accu(square(Y)));
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
                  bool is_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_impicit<double>(
      Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps,
      is_bias_last_row
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
                  bool is_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  // get arma matrices which share memory with R "float" matrices
  arma::fmat X = exctract_float_matrix(X_);
  arma::fmat Y = exctract_float_matrix(Y_);
  arma::fmat XtX = exctract_float_matrix(XtX_);
  return (double)als_impicit<float>(
      Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps,
      is_bias_last_row
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
                           bool is_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_explicit<double>(
      Conf, X, Y, lambda, n_threads, solver, cg_steps,
      with_biases, is_bias_last_row
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
                          bool is_bias_last_row) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  arma::fmat X = exctract_float_matrix(X_);
  arma::fmat Y = exctract_float_matrix(Y_);
  return (double)als_explicit<float>(
      Conf, X, Y, lambda, n_threads, solver, cg_steps,
      with_biases, is_bias_last_row
  );
}
