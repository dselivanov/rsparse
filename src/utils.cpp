#include <RcppArmadillo.h>
#include <queue>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Rcpp;

// Find top k elements and their indices on O(n * log (k)) time with heaps
// https://stackoverflow.com/a/38391603/1069256
// [[Rcpp::export]]
IntegerMatrix top_k_indices_byrow(const NumericMatrix &x, int k, int n_threads, Rcpp::Nullable<const arma::sp_mat> &not_recommend) {
  arma::sp_mat mat;
  int not_empty_filter_matrix = not_recommend.isNotNull();
  if(not_empty_filter_matrix)
    mat = as<arma::sp_mat>(not_recommend.get());

  int nc = x.ncol();
  int nr = x.nrow();
  IntegerMatrix res(nr, k);
  int *res_ptr = res.begin();
  NumericMatrix scores(nr, k);
  double *scores_ptr = scores.begin();
  const double *ptr = x.begin();
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(static)
  #endif
  for(int j = 0; j < nr; j++) {
    std::priority_queue< std::pair<double, int>, std::vector< std::pair<double, int> >, std::greater <std::pair<double, int> > > q;
    for (int i = 0; i < nc; ++i) {
      int ind = nr * i + j;
      double val = ptr[ind];
      double m_ji;
      if(not_empty_filter_matrix)
        m_ji = mat(j, i);
      else
        m_ji = 0;
      if(q.size() < k){
        if (m_ji == 0) q.push(std::pair<double, int>(val, i));
      } else if (q.top().first < val && m_ji == 0) {
        q.pop();
        q.push(std::pair<double, int>(val, i));
      }
    }
    for (int i = 0; i < k; ++i) {
      res_ptr[nr * (k - i - 1) + j] = q.top().second + 1;
      scores_ptr[nr * (k - i - 1) + j] = q.top().first;
      q.pop();
      // pathologic case
      // break if there were less than k predictions
      if(q.size() == 0) break;
    }
  }
  res.attr("scores") = scores;
  return(res);
}


// [[Rcpp::export]]
IntegerMatrix dotprod_top_k(const arma::mat &x, const arma::mat &y, int k, int n_threads, Rcpp::Nullable<const arma::sp_mat> &not_recommend) {
  arma::sp_mat mat;
  int not_empty_filter_matrix = not_recommend.isNotNull();
  if(not_empty_filter_matrix)
    mat = as<arma::sp_mat>(not_recommend.get());

  int nr = x.n_rows;
  int nc = y.n_cols;

  IntegerMatrix res(nr, k);
  int *res_ptr = res.begin();
  NumericMatrix scores(nr, k);
  double *scores_ptr = scores.begin();
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(static)
  #endif
  for(int j = 0; j < nr; j++) {
    arma::rowvec yvec = x.row(j) * y;
    std::priority_queue< std::pair<double, int>, std::vector< std::pair<double, int> >, std::greater <std::pair<double, int> > > q;
    for (int i = 0; i < nc; ++i) {
      // double val = as_scalar(x.row(j) * y.col(i));
      double val = arma::as_scalar(yvec.at(i));
      double m_ji;
      if(not_empty_filter_matrix)
        m_ji = mat(j, i);
      else
        m_ji = 0;
      if(q.size() < k){
        if (m_ji == 0) q.push(std::pair<double, int>(val, i));
      } else if (q.top().first < val && m_ji == 0) {
        q.pop();
        q.push(std::pair<double, int>(val, i));
      }
    }
    for (int i = 0; i < k; ++i) {
      res_ptr[nr * (k - i - 1) + j] = q.top().second + 1;
      scores_ptr[nr * (k - i - 1) + j] = q.top().first;
      q.pop();
      // pathologic case
      // break if there were less than k predictions
      if(q.size() == 0) break;
    }
  }
  res.attr("scores") = scores;
  return(res);
}
