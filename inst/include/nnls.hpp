#include <armadillo>

#define EPS 1e-16
#define RANDU_MAX 0.01

template <class T>
arma::Col<T> scd_ls_update(const arma::Mat<T> &WtW,
                   arma::Col<T> &mu,
                   arma::uword max_iter,
                   double rel_tol,
                   const arma::Col<T> &initial) {
  // Problem:  Aj = W * Hj
  // Method: sequential coordinate-wise descent
  // WtW = W^T W
  // WtAj = W^T Aj

  arma::Col<T> res = initial;
  const arma::Col<T> WtW_diag = WtW.diag();
  for (auto t = 0; t < max_iter; t++) {
    T rel_diff = 0;
    for (auto k = 0; k < WtW.n_cols; k++) {
      T current = res(k);
      auto update = current - mu(k) / WtW_diag.at(k);
      if(update < 0) update = 0;
      res(k) = update;
      if (update != current) {
        mu += (update - current) * WtW.col(k);
        auto diff = std::abs(current - update) / (std::abs(current) + EPS);
        if (diff > rel_diff) rel_diff = diff;
      }
    }
    if (rel_diff <= rel_tol) break;
  }
  return res;
}

template <class T>
arma::Col<T> c_nnls(const arma::Mat<T> &x,
                    const arma::Col<T> &y,
                    const arma::Col<T> &init,
                    arma::uword max_iter,
                    double rel_tol) {
  arma::Mat<T> Xt = x.t();

  arma::Mat<T> XtX = Xt * x;
  arma::Col<T> mu, sumW;

  // for stability: avoid divided by 0
  XtX.diag() += EPS;

  mu = XtX * init - Xt * y;
  const arma::Col<T> H = scd_ls_update<T>(XtX, mu, max_iter, rel_tol, init);

  return (H);
}
