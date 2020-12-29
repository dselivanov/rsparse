#include "rsparse.h"
// seed modes:
// keep_existing 1
// static_subset 2
// static_spread 3
// random_subset 4
// random_spread 5

// [[Rcpp::export]]
int arma_kmeans(const arma::dmat& x, const int k, const int seed_mode, const int n_iter,
                bool verbose, Rcpp::NumericMatrix& result) {
  arma::dmat result_map =
      arma::dmat(result.begin(), result.nrow(), result.ncol(), false, false);
  int status =
      arma::kmeans(result_map, x, k, arma::gmm_seed_mode(seed_mode), n_iter, verbose);
  return status;
}
