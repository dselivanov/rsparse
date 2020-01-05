#include "rsparse.h"
#include <queue>
#include <vector>

Rcpp::NumericMatrix NumericMatrixNA(int n, int m){
  Rcpp::NumericMatrix x(n, m) ;
  std::fill( x.begin(), x.end(), Rcpp::NumericVector::get_na() ) ;
  return x ;
}

Rcpp::IntegerMatrix IntegerMatrixNA(int n, int m){
  Rcpp::IntegerMatrix x(n, m) ;
  std::fill( x.begin(), x.end(), Rcpp::IntegerVector::get_na() ) ;
  return x ;
}

// Find top k elements (and their indices) of the dot-product of 2 matrices in O(n * log (k))
// https://stackoverflow.com/a/38391603/1069256
// [[Rcpp::export]]
Rcpp::IntegerMatrix top_product(const arma::mat &x, const arma::mat &y,
                          unsigned k, unsigned n_threads,
                          const Rcpp::S4 &not_recommend_r,
                          const Rcpp::IntegerVector &exclude) {
  std::unordered_set<int> exclude_set;
  for(Rcpp::IntegerVector::const_iterator it = exclude.begin(); it != exclude.end(); ++it) {
    exclude_set.insert( *it );
  }

  const dMappedCSR not_recommend = extract_mapped_csr(not_recommend_r);

  int not_empty_filter_matrix = not_recommend.nnz > 0;

  size_t nr = x.n_rows;
  size_t nc = y.n_cols;

  // init matrices with NA by default
  Rcpp::IntegerMatrix res = IntegerMatrixNA(nr, k);
  Rcpp::NumericMatrix scores = NumericMatrixNA(nr, k);

  arma::imat res_arma = arma::imat(res.begin(), nr, k, false, false);
  arma::dmat scores_arma = arma::dmat(scores.begin(), nr, k, false, false);

  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE)
  #endif
  for(size_t j = 0; j < nr; j++) {
    arma::uvec not_recommend_col_indices;

    if(not_empty_filter_matrix) {
      arma::uword p1 = not_recommend.row_ptrs[j];
      arma::uword p2 = not_recommend.row_ptrs[j + 1];
      not_recommend_col_indices = arma::uvec(&not_recommend.col_indices[p1], p2 - p1, false, true);
    }
    // points to current postion amoung indices which should be excluded for a given row
    size_t u = 0;

    arma::rowvec yvec = x.row(j) * y;

    std::priority_queue< std::pair<double, int>, std::vector< std::pair<double, int> >, std::greater <std::pair<double, int> > > q;
    // iterate through all columns and add insert top values in queue
    // also checks if current column should be excluded
    for (size_t i = 0; i < nc; ++i) {
      double val = arma::as_scalar(yvec(i));

      bool skip = false;
      // skip if column should be excluded for a given row
      if(not_empty_filter_matrix &&
         not_recommend_col_indices.size() > 0 &&
         u < not_recommend_col_indices.size()) {
          if(i == not_recommend_col_indices(u)) {
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
    // q_size always <= k
    arma::uword q_size = q.size();
    for (size_t i = 0; i < q_size; ++i) {
      // fill from the end because queue holds smallest element as top element
      res_arma(j, q_size - i - 1) = q.top().second + 1;
      scores_arma(j, q_size - i - 1) = q.top().first;
      q.pop();
    }
  }
  res.attr("scores") = scores;
  return(res);
}
