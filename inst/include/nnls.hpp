#include <armadillo>

#define TINY_NUM 1e-16
#define NNLM_REL_TOL 1e-8
#define MAX_ITER 500

template <class T>
void scd_ls_update(arma::subview_col<T> Hj,
                   const arma::Mat<T> &WtW,
                   arma::Col<T> &mu,
                   uint max_iter,
                   double rel_tol) {
  // Problem:  Aj = W * Hj
  // Method: sequential coordinate-wise descent when loss function = square error
  // WtW = W^T W
  // WtAj = W^T Aj

  auto WtW_diag = WtW.diag();
  for (auto t = 0; t < max_iter; t++) {
    T rel_err = 0;
    for (auto k = 0; k < WtW.n_cols; k++) {
      T current = Hj(k);
      auto update = current - mu(k) / WtW_diag.at(k);
      if(update < 0) update = 0;
      Hj(k) = update;
      if (update != current) {
        mu += (update - current) * WtW.col(k);
        auto current_err = std::abs(current - update) / (std::abs(current) + TINY_NUM);
        if (current_err > rel_err) rel_err = current_err;
      }
    }
    if (rel_err <= rel_tol) break;
  }
}

template <class T>
arma::Mat<T> c_nnls(const arma::Mat<T> &x,
                    const arma::Mat<T> &y,
                    uint max_iter,
                    double rel_tol) {
  arma::Mat<T> H(x.n_cols, y.n_cols, arma::fill::randu);
  arma::Mat<T> Wt = x.t();

  arma::Mat<T> WtW = Wt * Wt.t();
  arma::Col<T> mu, sumW;

  // for stability: avoid divided by 0
  WtW.diag() += TINY_NUM;

  for (unsigned int j = 0; j < y.n_cols; j++) {
    mu = WtW * H.col(j) - Wt * y.col(j);
    scd_ls_update<T>(H.col(j), WtW, mu, max_iter, rel_tol);
  }
  return (H);
}
