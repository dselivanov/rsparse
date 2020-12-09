#include <armadillo>

#define EPS 1e-16

template <class T>
arma::Col<T> scd_ls_update(const arma::Mat<T> &XtX,
                   arma::Col<T> &mu,
                   arma::uword max_iter,
                   double rel_tol,
                   const arma::Col<T> &initial) {
  arma::Col<T> res = initial;
  T rel_diff, old_value, new_value, diff;
  const arma::Col<T> XtX_diag = XtX.diag();
  for (auto t = 0; t < max_iter; t++) {
    rel_diff = 0;
    for (auto k = 0; k < XtX.n_cols; k++) {
      old_value = res(k);
      new_value = old_value - mu(k) / XtX_diag(k);
      if(new_value < 0) new_value = 0;
      diff = new_value - old_value;
      if (diff != 0) {
        res(k) = new_value;
        mu += diff * XtX.unsafe_col(k);
        auto step_err = std::abs(diff) / (std::abs(old_value) + EPS);
        if (step_err > rel_diff) rel_diff = step_err;
      }
    }
    if (rel_diff <= rel_tol) break;
  }
  return res;
}

template <class T>
arma::Col<T> c_nnls(const arma::Mat<T> &X,
                    const arma::Col<T> &y,
                    const arma::Col<T> &init,
                    arma::uword max_iter,
                    double rel_tol) {

  arma::Mat<T> Xt = X.t();
  arma::Mat<T> XtX = Xt * X;
  // for stability: avoid divided by 0
  XtX.diag() += EPS;
  arma::Col<T> mu = XtX * init - Xt * y;

  const arma::Col<T> H = scd_ls_update<T>(XtX, mu, max_iter, rel_tol, init);

  return (H);
}
