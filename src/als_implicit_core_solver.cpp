#define CHOLESKY 0
#define CONJUGATE_GRADIENT 1
#define TOL 1e-10

#include <RcppArmadillo.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif

#define GRAIN_SIZE 1
using namespace Rcpp;
using namespace RcppArmadillo;
using namespace arma;

template <class T>
arma::Col<T> chol_solver(const arma::Mat<T> &XtX,
                      const arma::Mat<T> &X_nnz,
                      const arma::Col<T> &confidence) {
  arma::Mat<T> inv = XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();
  arma::Mat<T> rhs = X_nnz * confidence;
  return solve(inv, rhs, solve_opts::fast );
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
T als_implicit_cpp(const arma::sp_mat& Conf,
                    arma::Mat<T>& X,
                    arma::Mat<T>& Y,
                    double lambda,
                    unsigned n_threads,
                    unsigned solver, unsigned cg_steps = 3) {

  arma::Mat<T> XtX = X * X.t();
  if(lambda > 0) {
    arma::Col<T> lambda_vec(X.n_rows);
    lambda_vec.fill(lambda);
    XtX += diagmat(lambda_vec);
  }

  T loss = 0;
  size_t nc = Conf.n_cols;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(size_t i = 0; i < nc; i++) {
    uint32_t p1 = Conf.col_ptrs[i];
    uint32_t p2 = Conf.col_ptrs[i + 1];
    // catch situation when some columns in matrix are empty, so p1 becomes equal to p2 or greater than number of columns
    if(p1 < p2) {
      arma::uvec idx = uvec(&Conf.row_indices[p1], p2 - p1);
      arma::Col<T> confidence = Col<T>(&Conf.values[p1], p2 - p1);
      arma::Mat<T> X_nnz = X.cols(idx);
      if(solver == CHOLESKY)
        Y.col(i) = chol_solver<T>(XtX, X_nnz, confidence);
      else if(solver == CONJUGATE_GRADIENT)
        Y.col(i) = cg_solver<T>(XtX, X_nnz, confidence, Y.col(i), cg_steps);
      else stop("Unknown solver code %d", solver);
      if(lambda >= 0)
        loss += accu(square( 1 - (Y.col(i).t() * X_nnz) ) * confidence);
    }
  }

  if(lambda > 0)
    loss += lambda * (accu(square(X)) + accu(square(Y)));
  return (loss / accu(Conf));
}


// [[Rcpp::export]]
double als_implicit(const arma::sp_mat& Conf,
                    arma::mat& X,
                    arma::mat& Y,
                    double lambda,
                    unsigned n_threads,
                    unsigned solver, unsigned cg_steps = 3) {
  return (double)als_implicit_cpp<double>(Conf, X, Y, lambda, n_threads, solver, cg_steps);
}

// [[Rcpp::export]]
double als_loss_explicit(const arma::sp_mat& mat, arma::mat& X, arma::mat& Y, double lambda, unsigned n_threads) {
  size_t nc = mat.n_cols;
  double loss = 0;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(size_t i = 0; i < nc; i++) {
    int p1 = mat.col_ptrs[i];
    int p2 = mat.col_ptrs[i + 1];
    for(int pp = p1; pp < p2; pp++) {
      size_t ind = mat.row_indices[pp];
      double diff = mat.values[pp] - as_scalar(Y.col(i).t() * X.col(ind));
      loss += diff * diff;
    }
  }
  if(lambda > 0)
    loss += lambda * (accu(square(X)) + accu(square(Y)));
  return loss / mat.n_nonzero;
}
