#include "MappedCSR.h"
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

dMappedCSR extract_mapped_csr(S4 input) {
  IntegerVector dim = input.slot("Dim");
  NumericVector rx = input.slot("x");
  uint32_t nrows = dim[0];
  uint32_t ncols = dim[1];
  IntegerVector rj = input.slot("j");
  IntegerVector rp = input.slot("p");
  return dMappedCSR(nrows, ncols, rx.length(), (uint32_t *)rj.begin(), (uint32_t *)rp.begin(), rx.begin());
}

// Find top k elements (and their indices) of the dot-product of 2 matrices in O(n * log (k))
// https://stackoverflow.com/a/38391603/1069256
// [[Rcpp::export]]
IntegerMatrix top_product(const arma::mat &x, const arma::mat &y,
                          unsigned k, unsigned n_threads,
                          S4 &not_recommend_r) {
  dMappedCSR not_recommend = extract_mapped_csr(not_recommend_r);

  int not_empty_filter_matrix = not_recommend.nnz > 0;

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
    arma::Col<uint32_t> not_recommend_row;

    if(not_empty_filter_matrix) {
      uint32_t p1 = not_recommend.p[j];
      uint32_t p2 = not_recommend.p[j + 1];
      not_recommend_row = arma::Col<uint32_t>(&not_recommend.j[p1], p2 - p1);
      // for(int l = 0; l < p2 - p1; l++) {
      //   not_recommend_row[l] = not_recommend.j[p1 + l];
      // }
    }
    size_t u = 0;
    arma::rowvec yvec = x.row(j) * y;

    std::priority_queue< std::pair<double, int>, std::vector< std::pair<double, int> >, std::greater <std::pair<double, int> > > q;
    for (size_t i = 0; i < nc; ++i) {
      double val = arma::as_scalar(yvec.at(i));

      bool skip = false;
      if(not_empty_filter_matrix && not_recommend_row.size() > 0) {
        if(i == not_recommend_row[u]) {
          skip = true;
          u++;
        }
      }

      if(q.size() < k) {
        if(!skip) q.push(std::pair<double, int>(val, i));
      } else if (q.top().first < val && !skip) {
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
