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
IntegerMatrix top_k_indices_byrow(const NumericMatrix &x, const arma::sp_mat &mat, int k, int n_threads) {
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

      double m_ji = mat(j, i);
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
