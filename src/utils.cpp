#include <RcppArmadillo.h>
#include <queue>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define GRAIN_SIZE 10

using namespace Rcpp;

// Find top k elements (and their indices) of the dot-product of 2 matrices in O(n * log (k))
// https://stackoverflow.com/a/38391603/1069256
// [[Rcpp::export]]
IntegerMatrix dotprod_top_k(const arma::mat &x, const arma::mat &y, unsigned k, unsigned n_threads, Rcpp::Nullable<const arma::sp_mat> &not_recommend) {
  arma::sp_mat mat;
  int not_empty_filter_matrix = not_recommend.isNotNull();
  if(not_empty_filter_matrix)
    mat = as<arma::sp_mat>(not_recommend.get());

  size_t nr = x.n_rows;
  size_t nc = y.n_cols;

  IntegerMatrix res(nr, k);
  int *res_ptr = res.begin();
  NumericMatrix scores(nr, k);
  double *scores_ptr = scores.begin();
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE)
  #endif
  for(size_t j = 0; j < nr; j++) {
    arma::rowvec yvec = x.row(j) * y;
    std::priority_queue< std::pair<double, int>, std::vector< std::pair<double, int> >, std::greater <std::pair<double, int> > > q;
    for (size_t i = 0; i < nc; ++i) {
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
    for (size_t i = 0; i < k; ++i) {
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
