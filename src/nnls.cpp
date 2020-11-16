#include <RcppArmadillo.h>
using namespace arma;

#ifdef _OPENMP
#include <omp.h>
#endif

#define TINY_NUM 1e-16
#define NNLM_REL_TOL 1e-8
#define MAX_ITER 500

void scd_ls_update(subview_col<double> Hj,
                  const mat & WtW,
                  vec & mu,
                  uint max_iter,
                  double rel_tol) {

  // Problem:  Aj = W * Hj
  // Method: sequential coordinate-wise descent when loss function = square error
  // WtW = W^T W
  // WtAj = W^T Aj

  double update;
  double error = 0;
  double rel_err;

  colvec WtW_diag = WtW.diag();
  for (auto t = 0; t < max_iter; t++) {
    rel_err = 0;
    for (auto k = 0; k < WtW.n_cols; k++) {
      double current = Hj(k);
      update = current - mu(k) / WtW_diag.at(k);
      if(update < 0) update = 0;

      if (update != current) {
        mu += (update - current) * WtW.col(k);
        auto current_err = std::abs(current - update) / (std::abs(current) + TINY_NUM);
        if (current_err > rel_err) rel_err = current_err;
        Hj(k) = update;
      }
    }
    if (rel_err <= rel_tol) break;
  }
}

void update(mat & H,
           const mat & Wt,
           const mat & A,
           uint max_iter,
           double rel_tol) {

  // A = W H, solve H
  // No missing in A, Wt = W^T

  int total_raw_iter = 0;

  mat WtW = Wt * Wt.t();
  colvec mu, sumW;

  // for stability: avoid divided by 0 in scd_ls, scd_kl
  WtW.diag() += TINY_NUM;

  for (unsigned int j = 0; j < A.n_cols; j++) {
    int iter = 0;
    mu = WtW * H.col(j) - Wt * A.col(j);
    scd_ls_update(H.col(j), WtW, mu, max_iter, rel_tol);
    total_raw_iter += iter;
  }
}

// [[Rcpp::export(rng=false)]]
arma::mat c_nnlm(const arma::mat & x,
                 const arma::mat & y,
                 uint max_iter,
                 double rel_tol) {
  mat coef(x.n_cols, y.n_cols);
  coef.randu();
  update(coef, x.t(), y, max_iter, rel_tol);
  return coef;
}
