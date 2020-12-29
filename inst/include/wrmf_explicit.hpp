#include "wrmf.hpp"
#include "wrmf_utils.hpp"
#include "nnls.hpp"

// arma::Mat<float> drop_row(const arma::Mat<float> &X_nnz, const bool drop_last);
// arma::Mat<double> drop_row(const arma::Mat<double> &X_nnz, const bool drop_last);

template <class T>
arma::Col<T> cg_solver_explicit(const arma::Mat<T> &X_nnz,
                                const arma::Col<T> &confidence,
                                const arma::Col<T> &x_old,
                                T lambda,
                                const arma::uword n_iter) {
  arma::Col<T> x = x_old;

  arma::Col<T> Ap;
  arma::Col<T> r = X_nnz * (confidence - (X_nnz.t() * x)) - lambda * x;
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = arma::dot(r, r);

  for(auto k = 0; k < n_iter; k++) {
    Ap = (X_nnz * (X_nnz.t() * p)) + lambda * p;
    alpha =  rsold / arma::dot(p, Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = arma::dot(r, r);
    if (rsnew < CG_TOL) break;
    p = r + p * (rsnew / rsold);
    rsold = rsnew;
  }
  return x;
}


template <class T>
T als_explicit(const dMappedCSC& Conf,
               arma::Mat<T>& X,
               arma::Mat<T>& Y,
               const double lambda,
               const unsigned n_threads,
               const unsigned solver,
               const unsigned cg_steps,
               const bool dynamic_lambda,
               const arma::Col<T>& cnt_X,
               const bool with_biases,
               const bool is_x_bias_last_row) {
  /* Note about biases:
   * For user factors, the first row will be set to all ones
   * to match with the item biases, and the calculated user biases will be in the
   * last row.
   * For item factors, the last row will be set to all ones to
   * mach with the user biases, and the calculated item biases will be in the
   * first row.
   */

  // if is_x_bias_last_row == true
  // X = [1, ..., x_bias]
  // Y = [y_bias, ..., 1]
  // if is_x_bias_last_row == false
  // X = [x_bias, ..., 1]
  // Y = [1, ..., y_bias]

  const arma::uword rank = X.n_rows;

  arma::Col<T> x_biases;

  if (with_biases) {
    if (is_x_bias_last_row) // last row
      x_biases = X.row(X.n_rows - 1).t();
    else // first row
      x_biases = X.row(0).t();
  }

  T loss = 0;
  size_t nc = Conf.n_cols;
#ifdef _OPENMP
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
#endif
  for(size_t i = 0; i < nc; i++) {
    arma::uword p1 = Conf.col_ptrs[i];
    arma::uword p2 = Conf.col_ptrs[i + 1];
    // catch situation when some columns in matrix are empty,
    // so p1 becomes equal to p2 or greater than number of columns
    if(p1 < p2) {
      const arma::uvec idx = arma::uvec(&Conf.row_indices[p1], p2 - p1, false, true);
      T lambda_use = lambda * (dynamic_lambda? static_cast<T>(p2-p1) : 1.);
      arma::Col<T> confidence = arma::conv_to< arma::Col<T> >::from(arma::vec(&Conf.values[p1], p2 - p1));
      arma::Mat<T> X_nnz = X.cols(idx);
      arma::Col<T> init = Y.col(i);
      // if is_x_bias_last_row == true
      // X_nnz = [1, ...]
      // if is_x_bias_last_row == false
      // X_nnz = [..., 1]
      if (with_biases) {
        X_nnz = drop_row<T>(X_nnz, is_x_bias_last_row);
        confidence -= x_biases(idx);
        init = drop_row<T>(init, !is_x_bias_last_row);
      }

      arma::Col<T> Y_new;
      // if is_x_bias_last_row == true
      // X_nnz = [1, ..., x_bias]
      // Y_new should be [y_bias, ...]
      // if is_x_bias_last_row == false
      // X_nnz = [x_bias, ..., 1]
      // Y_new should be [..., y_bias]
      if (solver == CONJUGATE_GRADIENT) {
        Y_new = cg_solver_explicit<T>(X_nnz, confidence, init, lambda_use, cg_steps);
      } else {
        arma::Mat<T> lhs = X_nnz * X_nnz.t();
        lhs.diag() += lambda_use;
        const arma::Mat<T> rhs = X_nnz * confidence;

        if (solver == CHOLESKY) { // CHOLESKY
          Y_new = solve(lhs, rhs, arma::solve_opts::fast );
        } else if (solver == SEQ_COORDINATE_WISE_NNLS) { // SEQ_COORDINATE_WISE_NNLS
          Y_new = c_nnls<T>(lhs, rhs, init, SCD_MAX_ITER, SCD_TOL);
        }
      }
      arma::Row<T> diff;

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
      diff = confidence.t() - (Y_new.t() * X_nnz);
      loss += arma::dot(diff, diff) + lambda_use * arma::dot(Y_new, Y_new);
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

  if(lambda > 0) {
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
      if (!dynamic_lambda)
        loss += lambda * accu(X_excl_ones % X_excl_ones);
      else {
        loss += lambda * accu((X_excl_ones % X_excl_ones) * cnt_X);
      }
    } else {
      if (!dynamic_lambda)
        loss += lambda * accu(X % X);
      else {
        loss += lambda * accu((X % X) * cnt_X);
      }
    }
  }
  return (loss / Conf.nnz);
}
