#include <RcppArmadillo.h>
#include <queue>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define GRAIN_SIZE 10

using namespace Rcpp;
using namespace RcppArmadillo;
using namespace arma;

// for sparse_hash_map < <uint32_t, uint32_t>, T >
#include <unordered_map>
#include <utility>
namespace std {
template <>
struct hash<std::pair<uint32_t, uint32_t> >
{
  inline uint64_t operator()(const std::pair<uint32_t, uint32_t>& k) const
  {
    return (uint64_t) k.first << 32 | k.second;
  }
};

}
// Find top k elements (and their indices) of the dot-product of 2 matrices in O(n * log (k))
// https://stackoverflow.com/a/38391603/1069256
// [[Rcpp::export]]
IntegerMatrix dotprod_top_k(const arma::mat &x, const arma::mat &y, unsigned k, unsigned n_threads,
                            Rcpp::Nullable<const arma::sp_mat> &not_recommend) {

  int not_empty_filter_matrix = not_recommend.isNotNull();
  const arma::sp_mat mat = (not_empty_filter_matrix) ? arma::sp_mat( as<arma::sp_mat>(not_recommend.get()) ) : arma::sp_mat();

  std::map< std::pair<uint32_t, uint32_t>, double > x_triplets;
  std::map < std::pair<uint32_t, uint32_t>, double > :: const_iterator triplet_iterator;
  for(sp_mat::const_iterator it = mat.begin(); it != mat.end(); ++it) {
    x_triplets[std::make_pair(it.row(), it.col())] = *it;
  }

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
      double m_ji = 0;

      if(not_empty_filter_matrix) {
        triplet_iterator = x_triplets.find(std::make_pair(j, i));
        if(triplet_iterator != x_triplets.end())
          m_ji = triplet_iterator->second;
      }

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




// [[Rcpp::export]]
NumericVector make_sparse_approximation(const arma::sp_mat& mat_template,
                       arma::mat& X, arma::mat& Y,
                       unsigned n_threads) {
  size_t nc = mat_template.n_cols;
  NumericVector approximated_values(mat_template.n_nonzero);
  double *ptr_approximated_values = approximated_values.begin();
  double loss = 0;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(size_t i = 0; i < nc; i++) {
    int p1 = mat_template.col_ptrs[i];
    int p2 = mat_template.col_ptrs[i + 1];
    if(p1 < p2) {
      arma::uvec idx = uvec(&mat_template.row_indices[p1], p2 - p1);
      arma::rowvec approximation(&ptr_approximated_values[p1], p2 - p1, false, true);
      approximation = Y.col(i).t() * X.cols(idx);
    }
  }
  return(approximated_values);
}
