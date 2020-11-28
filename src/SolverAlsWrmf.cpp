#include "rsparse.h"
#include "nnls.hpp"
#define IMPLICIT_FEEDBACK 0
#define EXPLICIT_FEEDBACK 1

template <class T>
arma::Col<T> cg_solver(const arma::Mat<T> &XtX,
                      const arma::Mat<T> &X_nnz,
                      const arma::Col<T> &confidence,
                      const arma::Col<T> &x_old,
                      const int n_iter) {
  arma::Col<T> x = x_old;
  arma::Col<T> confidence_1 = confidence - 1.0;

  arma::Col<T> Ap;
  arma::Col<T> r = X_nnz * (confidence - (confidence_1 % (X_nnz.t() * x))) - XtX * x;
  arma::Col<T> p = r;
  double rsold, rsnew, alpha;
  rsold = arma::dot(r, r);

  for(int k = 0; k < n_iter; k++) {
    Ap = XtX * p + X_nnz * (confidence_1 % (X_nnz.t() * p));
    alpha =  rsold / dot(p, Ap);
    x += alpha * p;
    r -= alpha * Ap;
    rsnew = dot(r, r);
    if (rsnew < TOL) break;
    p = r + p * (rsnew / rsold);
    rsold = rsnew;
  }
  return x;
}

template <class T>
T als_cpp(const dMappedCSC& Conf,
          arma::Mat<T>& X,
          arma::Mat<T>& Y,
          const arma::Mat<T>& XtX,
          double lambda,
          unsigned n_threads,
          unsigned solver,
          unsigned cg_steps = 3,
          bool non_negative = false,
          const arma::uword feedback =  IMPLICIT_FEEDBACK) {

  if (non_negative) solver = CHOLESKY;

  if(solver != CHOLESKY && solver != CONJUGATE_GRADIENT)
    Rcpp::stop("Unknown solver code %d", solver);

  T loss = 0;
  size_t nc = Conf.n_cols;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(size_t i = 0; i < nc; i++) {
    arma::uword p1 = Conf.col_ptrs[i];
    arma::uword p2 = Conf.col_ptrs[i + 1];
    // catch situation when some columns in matrix are empty, so p1 becomes equal to p2 or greater than number of columns
    if(p1 < p2) {
      auto idx = arma::uvec(&Conf.row_indices[p1], p2 - p1, false, true);
      const arma::Col<T> confidence = arma::conv_to< arma::Col<T> >::from(arma::vec(&Conf.values[p1], p2 - p1));
      const arma::Mat<T> X_nnz = X.cols(idx);
      if (feedback == IMPLICIT_FEEDBACK) {
        if(solver == CHOLESKY) {
          const arma::Mat<T> lhs = XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();
          const arma::Mat<T> rhs = X_nnz * confidence;
          if(non_negative) {
            Y.col(i) = c_nnls<T>(lhs, rhs, 10000, 1e-3);
          } else {
            Y.col(i) = solve(lhs, rhs, arma::solve_opts::fast );
          }
        } else { // CONJUGATE_GRADIENT
          Y.col(i) = cg_solver<T>(XtX, X_nnz, confidence, Y.col(i), cg_steps);
        }
        loss += dot(square( 1 - (Y.col(i).t() * X_nnz)), confidence);

      } else { //EXPLICIT_FEEDBACK
        // only support CHOLESKY for now
        arma::Mat<T> lhs = X_nnz * X_nnz.t();
        lhs.diag() += lambda;
        const arma::Mat<T> rhs = X_nnz * confidence;
        if(non_negative) {
          Y.col(i) = c_nnls<T>(lhs, rhs, 10000, 1e-3);
        } else {
          Y.col(i) = solve(lhs, rhs, arma::solve_opts::fast );
        }
        auto diff = confidence.t() - (Y.col(i).t() * X_nnz);
        loss += arma::dot(diff, diff);
      }
    } else {
      Y.col(i).zeros();
    }
  }

  if(lambda > 0)
    loss += lambda * (accu(square(X)) + accu(square(Y)));

  return (loss / Conf.nnz);
}


// [[Rcpp::export]]
double als_double(const Rcpp::S4 &m_csc_r,
                  arma::mat& X,
                  arma::mat& Y,
                  const arma::mat& XtX,
                  double lambda,
                  unsigned n_threads,
                  unsigned solver,
                  unsigned cg_steps,
                  bool non_negative,
                  const arma::uword feedback) {
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  return (double)als_cpp<double>(Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps, non_negative, feedback);
}

// [[Rcpp::export]]
double als_float( const Rcpp::S4 &m_csc_r,
                  Rcpp::S4 &XR,
                  Rcpp::S4 & YR,
                  Rcpp::S4 &XtXR,
                  double lambda,
                  unsigned n_threads,
                  unsigned solver,
                  unsigned cg_steps,
                  bool non_negative,
                  const arma::uword feedback) {
  //#ifdef SINGLE_PRECISION_LAPACK_AVAILABLE
  const dMappedCSC Conf = extract_mapped_csc(m_csc_r);
  Rcpp::IntegerMatrix XRM = XR.slot("Data");
  Rcpp::IntegerMatrix YRM = YR.slot("Data");
  Rcpp::IntegerMatrix XtXRM = XtXR.slot("Data");
  float * x_ptr = reinterpret_cast<float *>(&XRM[0]);
  float * y_ptr = reinterpret_cast<float *>(&YRM[0]);
  float * xtx_ptr = reinterpret_cast<float *>(&XtXRM[0]);
  arma::fmat X = arma::fmat(x_ptr, XRM.nrow(), XRM.ncol(), false, true);
  arma::fmat Y = arma::fmat(y_ptr, YRM.nrow(), YRM.ncol(), false, true);
  arma::fmat XtX = arma::fmat(xtx_ptr, XtXRM.nrow(), XtXRM.ncol(), false, true);
  return (double)als_cpp<float>(Conf, X, Y, XtX, lambda, n_threads, solver, cg_steps, non_negative, feedback);
}
