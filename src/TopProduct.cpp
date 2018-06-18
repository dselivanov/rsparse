#include "rsparse.h"
#include <queue>
#include <vector>

// Find top k elements (and their indices) of the dot-product of 2 matrices in O(n * log (k))
// https://stackoverflow.com/a/38391603/1069256
// [[Rcpp::export]]
Rcpp::IntegerMatrix top_product(const arma::mat &x, const arma::mat &y,
                          unsigned k, unsigned n_threads,
                          Rcpp::S4 &not_recommend_r,
                          const Rcpp::IntegerVector &exclude) {
  std::unordered_set<int> exclude_set;
  for(Rcpp::IntegerVector::const_iterator it = exclude.begin(); it != exclude.end(); ++it) {
    exclude_set.insert( *it );
  }

  dMappedCSR not_recommend = extract_mapped_csr(not_recommend_r);

  int not_empty_filter_matrix = not_recommend.nnz > 0;

  size_t nr = x.n_rows;
  size_t nc = y.n_cols;

  Rcpp::IntegerMatrix res(nr, k);
  int *res_ptr = res.begin();
  Rcpp::NumericMatrix scores(nr, k);
  double *scores_ptr = scores.begin();
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE)
  #endif
  for(size_t j = 0; j < nr; j++) {
    arma::Col<uint32_t> not_recommend_col_indices;

    if(not_empty_filter_matrix) {
      uint32_t p1 = not_recommend.row_ptrs[j];
      uint32_t p2 = not_recommend.row_ptrs[j + 1];
      not_recommend_col_indices = arma::Col<uint32_t>(&not_recommend.col_indices[p1], p2 - p1);
    }
    // points to current postion amoung indices which should be excluded for a given row
    size_t u = 0;

    arma::rowvec yvec = x.row(j) * y;

    std::priority_queue< std::pair<double, int>, std::vector< std::pair<double, int> >, std::greater <std::pair<double, int> > > q;
    // iterate through all columns and add insert top values in queue
    // also checks if current column should be excluded
    for (size_t i = 0; i < nc; ++i) {
      double val = arma::as_scalar(yvec.at(i));

      bool skip = false;
      // skip if column should be excluded for a given row
      if(not_empty_filter_matrix && not_recommend_col_indices.size() > 0) {
        if(i == not_recommend_col_indices[u]) {
          skip = true;
          u++;
        }
      }
      // skip if column excluded globally
      // add + 1 because inidices in R start from 1
      if(exclude_set.find(i + 1) != exclude_set.end()) skip = true;

      if(q.size() < k) {
        if(!skip) q.push(std::pair<double, int>(val, i));
      } else if (q.top().first < val && !skip) {
        q.pop();
        q.push(std::pair<double, int>(val, i));
      }
    }
    for (size_t i = 0; i < k; ++i) {
      int ind = nr * (k - i - 1) + j;
      res_ptr[ind] = q.top().second + 1;
      scores_ptr[ind] = q.top().first;
      q.pop();
      // pathologic case
      // break if there were less than k predictions
      if(q.size() == 0) break;
    }
  }
  res.attr("scores") = scores;
  return(res);
}
