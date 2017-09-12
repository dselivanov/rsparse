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

arma::vec chol_solver(const arma::mat &XtX,
                      const arma::mat &X_nnz,
                      const arma::vec &confidence) {
  arma::mat inv = XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();
  arma::mat rhs = X_nnz * confidence;
  return solve(inv, rhs, solve_opts::fast );
}

inline arma::vec cg_solver(const arma::mat &XtX,
                      const arma::mat &X_nnz,
                      const arma::vec &confidence,
                      const arma::vec &x_old,
                      const int n_iter) {
  arma::colvec x = x_old;
  arma::vec confidence_1 = confidence - 1;

  arma::mat Ap;
  arma::vec r = X_nnz * (confidence - (confidence_1 % (X_nnz.t() * x))) - XtX * x;
  arma::vec p = r;
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

// [[Rcpp::export]]
double als_implicit(const arma::sp_mat& Conf,
                    arma::mat& X,
                    arma::mat& Y,
                    double lambda,
                    int n_threads,
                    int solver, int cg_steps = 3) {

  arma::mat XtX = X * X.t();
  if(lambda > 0) {
    arma::vec lambda_vec(X.n_rows);
    lambda_vec.fill(lambda);
    XtX += diagmat(lambda_vec);
  }

  double loss = 0;
  int nc = Conf.n_cols;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(int i = 0; i < nc; i++) {
    int p1 = Conf.col_ptrs[i];
    int p2 = Conf.col_ptrs[i + 1];
    // catch situation when some columns in matrix are empty, so p1 becomes equal to p2 or greater than number of columns
    if(p1 < p2) {
      arma::uvec idx = uvec(&Conf.row_indices[p1], p2 - p1);
      arma::vec confidence = vec(&Conf.values[p1], p2 - p1);
      arma::mat X_nnz = X.cols(idx);
      if(solver == CHOLESKY)
        Y.col(i) = chol_solver(XtX, X_nnz, confidence);
      else if(solver == CONJUGATE_GRADIENT)
        Y.col(i) = cg_solver(XtX, X_nnz, confidence, Y.col(i), cg_steps);
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
double als_loss_explicit(const arma::sp_mat& mat, arma::mat& X, arma::mat& Y, double lambda, int n_threads) {
  int nc = mat.n_cols;
  double loss = 0;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(int i = 0; i < nc; i++) {
    int p1 = mat.col_ptrs[i];
    int p2 = mat.col_ptrs[i + 1];
    if(p1 < p2) {
      arma::uvec idx = uvec(&mat.row_indices[p1], p2 - p1);
      arma::vec rating = vec(&mat.values[p1], p2 - p1);
      loss += accu(square( rating.t() - (Y.col(i).t() * X.cols(idx))));
    }
  }
  if(lambda > 0)
    loss += lambda * (accu(square(X)) + accu(square(Y)));
  return loss / accu(mat);
}
