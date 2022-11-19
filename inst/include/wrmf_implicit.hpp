#include "nnls.hpp"
#include "wrmf.hpp"
#include "wrmf_utils.hpp"

// arma::Mat<float> drop_row(const arma::Mat<float> &X_nnz, const bool drop_last);
// arma::Mat<double> drop_row(const arma::Mat<double> &X_nnz, const bool drop_last);

template <class T>
arma::Col<T> cg_solver_implicit(const arma::Mat<T>& X_nnz, const arma::Col<T>& confidence,
                                const arma::Col<T>& x_old, const arma::uword n_iter,
                                const arma::Mat<T>& XtX) {
  arma::Col<T> x = x_old;
  const arma::Col<T> confidence_1 = confidence - 1.0;

  arma::Col<T> Ap;
  arma::Col<T> r = X_nnz * (confidence - (confidence_1 % (X_nnz.t() * x))) - XtX * x;
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = arma::dot(r, r);

  for (auto k = 0; k < n_iter; k++) {
    Ap = XtX * p + X_nnz * (confidence_1 % (X_nnz.t() * p));
    alpha = rsold / dot(p, Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = dot(r, r);
    if (rsnew < CG_TOL) break;
    p = r + p * (rsnew / rsold);
    rsold = rsnew;
  }
  return x;
}

/* FIXME: 'global_bias_base' used like this has very poor numerical precision and ends up making things worse */
template <class T>
arma::Col<T> cg_solver_implicit_global_bias(const arma::Mat<T>& X_nnz, const arma::Col<T>& confidence,
                                            const arma::Col<T>& x_old, const arma::uword n_iter,
                                            const arma::Mat<T>& XtX,   const arma::Col<T> global_bias_base,
                                            T global_bias) {
  arma::Col<T> x = x_old;
  const arma::Col<T> confidence_1 = confidence - 1.0;

  arma::Col<T> Ap;
  arma::Col<T> r = X_nnz * (confidence - (confidence_1 % (X_nnz.t() * x + global_bias))) - XtX * x + global_bias_base;
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = arma::dot(r, r);

  for (auto k = 0; k < n_iter; k++) {
    Ap = XtX * p + X_nnz * (confidence_1 % (X_nnz.t() * p));
    alpha = rsold / dot(p, Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = dot(r, r);
    if (rsnew < CG_TOL) break;
    p = r + p * (rsnew / rsold);
    rsold = rsnew;
  }
  return x;
}

template <class T>
arma::Col<T> cg_solver_implicit_user_item_bias(const arma::Mat<T>& X_nnz, const arma::Col<T>& confidence,
                                               const arma::Col<T>& x_old, const arma::uword n_iter,
                                               const arma::Mat<T>& XtX,   const arma::Col<T> &rhs_init,
                                               const arma::Col<T> &x_biases, T global_bias) {
  arma::Col<T> x = x_old;
  const arma::Col<T> confidence_1 = confidence - 1.0;

  arma::Col<T> Ap;
  arma::Col<T> r = X_nnz * (confidence - (confidence_1 % (X_nnz.t() * x + x_biases + global_bias)))
                    - XtX * x + rhs_init;
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = arma::dot(r, r);

  for (auto k = 0; k < n_iter; k++) {
    Ap = XtX * p + X_nnz * (confidence_1 % (X_nnz.t() * p));
    alpha = rsold / dot(p, Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = dot(r, r);
    if (rsnew < CG_TOL) break;
    p = r + p * (rsnew / rsold);
    rsold = rsnew;
  }
  return x;
}

template <class T>
T als_implicit(const dMappedCSC& Conf, arma::Mat<T>& X, arma::Mat<T>& Y,
               const arma::Mat<T>& XtX, double lambda, int n_threads,
               const unsigned int solver, unsigned int cg_steps, const bool with_biases,
               const bool is_x_bias_last_row, double global_bias,
               arma::Col<T> &global_bias_base, const bool initialize_bias_base) {
  // if is_x_bias_last_row == true
  // X = [1, ..., x_bias]
  // Y = [y_bias, ..., 1]
  // if is_x_bias_last_row == false
  // X = [x_bias, ..., 1]
  // Y = [1, ..., y_bias]

  const arma::uword rank = X.n_rows;

  arma::Col<T> x_biases;
  arma::Mat<T> rhs_init;

  if (global_bias < std::sqrt(std::numeric_limits<T>::epsilon()))
    global_bias = 0;

  if (global_bias && initialize_bias_base && !with_biases)
    global_bias_base = arma::sum(X, 1) * (-global_bias);

  if (with_biases) {
    if (is_x_bias_last_row)  // last row
      x_biases = X.row(X.n_rows - 1).t();
    else  // first row
      x_biases = X.row(0).t();

    // if we model bias then `rhs = X * C_u * (p_u - x_biases)`
    // where
    // X - item embeddings matrix
    // `C_u` is a diagonal matrix of confidences (n_item*n_item)
    // C_u has a form: `C_ui = 1 + f(r_ui)`
    // For missing entries C_u = 1
    // `p` is an indicator function:
    // - 0 if user-item interaction missing
    // - 1 if user-item interaction is present

    // we can rewrite it as
    // rhs = X * 1 * (0 - x_biases) + X * (1 + f(r_ui)) * (1 - x_biases)

    // we know that most of the interactions are missing (p=0)
    // so we can pre-compute `rhs_init` for all p=0:
    // `rhs_init = X * 1 * (0 - x_biases)`
    //
    // and then for each user we can calculate `rhs` using
    // small a update from`rhs_init`.
    // For non-missing interactions (p=1)
    // rhs_user = rhs_init - \
    //    X_nnz_user * 1 * (0 - x_biases_nnz_user) +
    //    X_nnz_user * C_nnz_user * (1 - x_biases_nnz_user)

    if (!global_bias) {
      // here we do following:
      // drop row with "ones" placeholder for convenient ALS form
      rhs_init = drop_row<T>(X, is_x_bias_last_row);
      // p = 0
      // C = 1 (so we omit multiplication on eye matrix)
      // rhs = X * eye * (0 - x_biases) = -X * x_biases
      rhs_init *= -x_biases;
    } else {
      rhs_init = - (drop_row<T>(X, is_x_bias_last_row) * (x_biases + global_bias));
    }
  } else if (global_bias) {
    rhs_init = arma::Mat<T>(&global_bias_base[0], rank - (int)with_biases, 1, false, true);
  }


  double loss = 0;
  size_t nc = Conf.n_cols;
#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads)
#endif
{
  arma::Mat<T> X_nnz;
  arma::Mat<T> X_nnz_t;
  arma::Col<T> init;
  arma::Col<T> Y_new;
  arma::Mat<T> rhs;

#ifdef _OPENMP
#pragma omp for schedule(dynamic) reduction(+:loss)
#endif
  for (size_t i = 0; i < nc; i++) {
    arma::uword p1 = Conf.col_ptrs[i];
    arma::uword p2 = Conf.col_ptrs[i + 1];
    // catch situation when some columns in matrix are empty, so p1 becomes equal to p2 or
    // greater than number of columns
    if (with_biases || global_bias || p1 < p2) {
      const arma::uvec idx = arma::uvec(&Conf.row_indices[p1], p2 - p1, false, true);
      arma::Col<T> confidence =
          arma::conv_to<arma::Col<T> >::from(arma::vec(&Conf.values[p1], p2 - p1));
      X_nnz = X.cols(idx);
      // if is_x_bias_last_row == true
      // X_nnz = [1, ...]
      // if is_x_bias_last_row == false
      // X_nnz = [..., 1]
      if (with_biases) {
        X_nnz = drop_row<T>(X_nnz, is_x_bias_last_row);
        // init = drop_row<T>(init, !is_x_bias_last_row);
      }

      if (solver == CONJUGATE_GRADIENT) {
        init = Y.col(i);
        if (!with_biases && !global_bias)
          Y_new = cg_solver_implicit<T>(X_nnz, confidence, init, cg_steps, XtX);
        else if (with_biases) {
          init = drop_row<T>(init, !is_x_bias_last_row);
          Y_new = cg_solver_implicit_user_item_bias<T>(X_nnz, confidence, init, cg_steps, XtX,
                                                       rhs_init, x_biases(idx), global_bias);
        } else {
          Y_new = cg_solver_implicit_global_bias<T>(X_nnz, confidence, init, cg_steps, XtX,
                                                    rhs_init, global_bias);
        }
      } else {
        const arma::Mat<T> lhs =
          XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();

        if (with_biases) {
          // now we need to update rhs with rhs_init and take into account
          // items with interactions (p=1)

          // first we reset contributions from items
          // where we considered p=0, but actually p=1 during rhs_init calculation

          // rhs = rhs_init + X_nnz * x_biases(idx); // rhs_init - (X_nnz * (0 -
          // x_biases(idx)))
          // // element-wise row multiplication is equal
          // // to multiplying on diagonal matrix
          // rhs += (X_nnz.each_row() % confidence.t()) * \
          //       // mult by (1 - biases)
          //       (1 - x_biases(idx));

          // expression above can be simplified further:
          rhs = rhs_init + X_nnz * (confidence - x_biases(idx) % (confidence - 1));

        } else if (global_bias) {
          rhs = X_nnz * confidence + rhs_init;
        } else {
          rhs = X_nnz * confidence;
        }
        if (solver == SEQ_COORDINATE_WISE_NNLS) {
          Y_new = c_nnls<T>(lhs, rhs, init, SCD_MAX_ITER, SCD_TOL);
        } else {  // CHOLESKY
          Y_new = solve(lhs, rhs, arma::solve_opts::fast + arma::solve_opts::likely_sympd);
        }
      }

      if (with_biases) {
        if (is_x_bias_last_row) {
          // X_nnz = [1, ..., x_bias]
          // Y_new should be [y_bias, ...]
          // Y.col(i) should be [y_bias, ..., 1]
          Y.unsafe_col(i).head(rank - 1) = Y_new;

        } else {
          // X_nnz = [x_bias, ..., 1]
          // Y_new should be [..., y_bias]
          // Y.col(i) should be [1, ..., y_bias]
          Y.unsafe_col(i).tail(rank - 1) = Y_new;
        }
      } else {
        Y.unsafe_col(i) = Y_new;
      }

      if (p1 == p2)
        loss += lambda * arma::dot(Y_new, Y_new);
      else if (!global_bias && !with_biases)
        loss += dot(square(1 - (Y_new.t() * X_nnz)), confidence) +
                lambda * arma::dot(Y_new, Y_new);
      else if (global_bias && !with_biases)
        loss += dot(square((1 - global_bias) - (Y_new.t() * X_nnz)), confidence) +
                lambda * arma::dot(Y_new, Y_new);
      else if (!global_bias && with_biases)
        loss += dot(square(1 - (Y_new.t() * X_nnz) - x_biases(idx).t()), confidence) +
                lambda * arma::dot(Y_new, Y_new);
      else
        loss += dot(square((1 - global_bias) - (Y_new.t() * X_nnz) - x_biases(idx).t()), confidence) +
                lambda * arma::dot(Y_new, Y_new);

    } else {
      if (with_biases) {
        const arma::Col<T> z(rank - 1, arma::fill::zeros);
        if (is_x_bias_last_row) {
          Y.unsafe_col(i).head(rank - 1) = z;
        } else {
          Y.unsafe_col(i).tail(rank - 1) = z;
        }
      } else {
        Y.unsafe_col(i).zeros();
      }
    }
  }
}
  if (lambda > 0) {
    if (with_biases) {
      // lambda applied to all learned parameters:
      // embeddings and biases
      // so we select all rows excluding dummy ones
      // if is_x_bias_last_row == true
      // X = [1, ..., x_bias]
      // Y = [y_bias, ..., 1]
      bool is_drop_last_x = !is_x_bias_last_row;
      const arma::Mat<T> X_excl_ones = drop_row<T>(X, is_drop_last_x);
      // as per arma docs "multiply-and-accumulate"
      // should should be translated
      // into efficient MKL/OpenBLAS calls
      loss += lambda * accu(X_excl_ones % X_excl_ones);
    } else {
      loss += lambda * accu(X % X);
    }
  }
  return (loss / Conf.nnz);
}
