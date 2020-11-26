#include <armadillo>

#define TINY_NUM 1e-16

template <class T>
arma::Col<T> scd_ls_update(const arma::Mat<T> &WtW,
                   arma::Col<T> &mu,
                   arma::uword max_iter,
                   double rel_tol,
                   const arma::Col<T> &initial) {
  // Problem:  Aj = W * Hj
  // Method: sequential coordinate-wise descent when loss function = square error
  // WtW = W^T W
  // WtAj = W^T Aj

  arma::Col<T> res = initial;
  auto WtW_diag = WtW.diag();
  for (auto t = 0; t < max_iter; t++) {
    T rel_err = 0;
    for (auto k = 0; k < WtW.n_cols; k++) {
      T current = res(k);
      auto update = current - mu(k) / WtW_diag.at(k);
      if(update < 0) update = 0;
      res(k) = update;
      if (update != current) {
        mu += (update - current) * WtW.col(k);
        auto current_err = std::abs(current - update) / (std::abs(current) + TINY_NUM);
        if (current_err > rel_err) rel_err = current_err;
      }
    }
    if (rel_err <= rel_tol) break;
  }
  return res;
}

template <class T>
arma::Mat<T> c_nnls(const arma::Mat<T> &x,
                    const arma::Mat<T> &y,
                    arma::uword max_iter,
                    double rel_tol) {
  arma::Mat<T> H(x.n_cols, y.n_cols, arma::fill::randu);
  arma::Mat<T> Wt = x.t();

  arma::Mat<T> WtW = Wt * x;
  arma::Col<T> mu, sumW;

  // for stability: avoid divided by 0
  WtW.diag() += TINY_NUM;

  for (auto j = 0; j < y.n_cols; j++) {
    mu = WtW * H.col(j) - Wt * y.col(j);
    H.col(j) = scd_ls_update<T>(WtW, mu, max_iter, rel_tol, H.col(j));
  }
  return (H);
}
