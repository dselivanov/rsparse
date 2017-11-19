#include <RcppArmadillo.h>
#include <queue>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#define GRAIN_SIZE 10

#define CSC 1
#define CSR 2

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
NumericVector cpp_make_sparse_approximation(const S4 &mat_template,
                                            arma::mat& X,
                                            arma::mat& Y,
                                            int sparse_matrix_type,
                                            unsigned n_threads) {
  IntegerVector rp = mat_template.slot("p");
  int* p = rp.begin();
  IntegerVector rj;
  if(sparse_matrix_type == CSR) {
    rj = mat_template.slot("j");
  } else if(sparse_matrix_type == CSC) {
    rj = mat_template.slot("i");
  } else
    ::Rf_error("make_sparse_approximation_csr doesn't know sparse matrix type. Should be CSC=1 or CSR=2");

  uint32_t* j = (uint32_t *)rj.begin();
  IntegerVector dim = mat_template.slot("Dim");

  size_t nr = dim[0];
  size_t nc = dim[1];
  int N;
  if(sparse_matrix_type == CSR)
    N = nr;
  else
    N = nc;

  NumericVector approximated_values(rj.length());

  double *ptr_approximated_values = approximated_values.begin();
  double loss = 0;
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE) reduction(+:loss)
  #endif
  for(size_t i = 0; i < N; i++) {
    int p1 = p[i];
    int p2 = p[i + 1];
    for(int pp = p1; pp < p2; pp++) {
      uint64_t ind = (size_t)j[pp];
      if(sparse_matrix_type == CSR)
        ptr_approximated_values[pp] = as_scalar(X.col(i).t() * Y.col(ind));
      else
        ptr_approximated_values[pp] = as_scalar(Y.col(i).t() * X.col(ind));
    }

  }
  return(approximated_values);
}

// [[Rcpp::export]]
List  arma_svd_econ(const arma::mat& X) {
  int k = std::min(X.n_rows, X.n_cols);
  NumericMatrix UR(X.n_rows, k);
  NumericMatrix VR(X.n_cols, k);
  NumericVector dR(k);
  arma::mat U(UR.begin(), UR.nrow(), UR.ncol(),  false, true);
  arma::mat V(VR.begin(), VR.nrow(), VR.ncol(),  false, true);
  arma::vec d(dR.begin(), dR.size(),  false, true);
  int status = svd_econ(U, d, V, X);
  if(!status)
    ::Rf_error("arma::svd_econ failed");
  return(List::create(_["d"] = dR, _["u"] = UR, _["v"] = VR));
}
