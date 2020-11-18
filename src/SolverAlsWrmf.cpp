#include "rsparse.h"
#include "nnls.hpp"

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

  // arma::Mat<T> XtX = X * X.t();
  // if(lambda > 0) {
  //   arma::Col<T> lambda_vec(X.n_rows);
  //   lambda_vec.fill(lambda);
  //   XtX += diagmat(lambda_vec);
  // }

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
                    Rcpp::S4 & YR,
                    Rcpp::S4 &XtXR,
                    double lambda,
                    unsigned n_threads,
                    unsigned solver,
                    unsigned cg_steps = 3,
                    bool non_negative = false) {
  //#ifdef SINGLE_PRECISION_LAPACK_AVAILABLE
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
double als_loss_explicit(const Rcpp::S4 &m_csc_r, arma::mat& X, arma::mat& Y, double lambda, unsigned n_threads) {
  dMappedCSC mat = extract_mapped_csc(m_csc_r);
  size_t nc = mat.n_cols;
  double loss = 0;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(size_t i = 0; i < nc; i++) {
    arma::colvec y_i = Y.col(i);
    auto p1 = mat.col_ptrs[i];
    auto p2 = mat.col_ptrs[i + 1];
    double diff = 0;
    for(int pp = p1; pp < p2; pp++) {
      auto j = mat.row_indices[pp];
      diff = mat.values[pp] - arma::dot(y_i, X.col(j));
      loss += diff * diff;
    }
  }
  if(lambda > 0)
    loss += lambda * (accu(square(X)) + accu(square(Y)));
  return loss / mat.nnz;
}
