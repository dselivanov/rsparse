#include <RcppArmadillo.h>
#include <omp.h>
#define GRAIN_SIZE 100
using namespace Rcpp;
using namespace RcppArmadillo;
using namespace arma;

//' @export
// [[Rcpp::export]]
void als_implicit(const S4 &mat, arma::mat& X, arma::mat& XtX, arma::mat& Y, int nth) {
  IntegerVector dims = mat.slot("Dim");
  int ncols = dims[1];
  arma::uvec CUI_I = Rcpp::as<arma::uvec>(mat.slot("i"));
  arma::uvec CUI_P = Rcpp::as<arma::uvec>(mat.slot("p"));
  arma::vec CUI_X  = Rcpp::as<arma::vec>(mat.slot("x"));

  #pragma omp parallel for num_threads(nth) schedule(dynamic, GRAIN_SIZE)
  for(int i = 0; i < ncols; i++) {
    int p1 = CUI_P[i];
    int p2 = CUI_P[i + 1] - 1;
    // catch situation when last column in matrix are empty, so p1 becomes larger than number of columns
    if(p1 <= p2) {
      arma::uvec idx = CUI_I.subvec( p1, p2 );
      arma::mat X_nnz = X.cols(idx);

      arma::vec confidence = CUI_X.subvec( p1, p2 );
      arma::mat inv = XtX + X_nnz.each_row() % (confidence.t() - 1) * X_nnz.t();
      arma::mat rhs = X_nnz * confidence;

      Y.col(i) = solve(inv, rhs, solve_opts::fast );
      // arma::mat CF = chol(inv);
      // arma::vec y = solve( trimatu(CF), rhs, solve_opts::fast);
      // Y.col(i) = solve(trimatu(CF.t()), y, solve_opts::fast);
    }
  }
}
