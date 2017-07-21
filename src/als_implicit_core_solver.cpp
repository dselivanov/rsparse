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

arma::vec chol_solver(const arma::sp_mat& Conf,
                      const arma::mat& X,
                      const arma::mat& XtX,
                      const arma::mat& Y,
                      arma::uvec idx,
                      arma::vec confidence) {
  arma::mat X_nnz = X.cols(idx);
  arma::mat inv = XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();
  arma::mat rhs = X_nnz * confidence;
  return solve(inv, rhs, solve_opts::fast );
}

arma::vec cg_solver(const arma::sp_mat& Conf,
                      const arma::mat& X,
                      const arma::mat& XtX,
                      const arma::uvec &idx,
                      const arma::vec &confidence,
                      const arma::vec &x_old,
                      int n_iter) {
  int DEBUG = 0;
  arma::colvec x = x_old;
  arma::mat X_nnz = X.cols(idx);
  arma::mat X_nnz_t = X_nnz.t();
  arma::vec confidence_1 = confidence - 1;

  arma::mat Ap;
  arma::vec r = X_nnz * confidence - XtX * x - X_nnz * (confidence_1 % (X_nnz_t * x));
  arma::vec p = r;
  double rsold, rsnew, alpha;

  rsold = as_scalar(r.t() * r);

  for(int k = 0; k < n_iter; k++) {
    Ap = XtX * p + X_nnz * (confidence_1 % (X_nnz_t * p));
    alpha =  rsold / as_scalar(p.t() * Ap);

    x += alpha * p;
    r -= alpha * Ap;
    rsnew = as_scalar(r.t() * r);

    if (rsnew < TOL) break;

    p = r + p * (rsnew / rsold);
    // if(DEBUG) printf("k = %d, alpha = %.8f, rsnew = %.3f, rsold = %.3f\n", k, alpha, rsnew, rsold);
    rsold = rsnew;
  }
  // if(DEBUG) printf("----------------------------------------\n");
  return x;
}

// [[Rcpp::export]]
void als_implicit(const arma::sp_mat& Conf, arma::mat& X, arma::mat& XtX, arma::mat& Y, int n_threads,
                  int solver, int cg_steps = 3) {
  int nc = Conf.n_cols;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE)
  #endif
  for(int i = 0; i < nc; i++) {
    int p1 = Conf.col_ptrs[i];
    int p2 = Conf.col_ptrs[i + 1];
    // catch situation when some columns in matrix are empty, so p1 becomes equal to p2 or greater than number of columns
    if(p1 < p2) {
      arma::uvec idx = uvec(&Conf.row_indices[p1], p2 - p1);
      arma::vec confidence = vec(&Conf.values[p1], p2 - p1);
      if(solver == CHOLESKY)
        Y.col(i) = chol_solver(Conf, X, XtX, Y, idx, confidence);
      else if(solver == CONJUGATE_GRADIENT)
        Y.col(i) = cg_solver(Conf, X, XtX, idx, confidence, Y.col(i), cg_steps);
      else stop("Unknown solver code %d", solver);
    }
  }
}

// [[Rcpp::export]]
double als_loss(const arma::sp_mat& mat, arma::mat& X, arma::mat& Y, double lambda, int feedback, int n_threads) {
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
      arma::vec true_val = vec(&mat.values[p1], p2 - p1);
      arma::mat user_i = X.cols(idx);
      if(feedback == 1) {
        // implicit feedback: true_val = confidence
        loss += as_scalar((square( 1 - (Y.col(i).t() * user_i) ) * true_val));
      } else if(feedback == 2) {
        // explicit feedback: true_val = rating
        loss += as_scalar(square( true_val.t() - (Y.col(i).t() * user_i) ));
      }
    }
  }
  if(lambda > 0)
    loss += lambda * (accu(square(X)) + accu(square(Y)));
  return loss / accu(mat);
}
