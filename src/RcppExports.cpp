// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// als_implicit
void als_implicit(const arma::sp_mat& mat, arma::mat& X, arma::mat& XtX, arma::mat& Y, int n_threads);
RcppExport SEXP reco_als_implicit(SEXP matSEXP, SEXP XSEXP, SEXP XtXSEXP, SEXP YSEXP, SEXP n_threadsSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const arma::sp_mat& >::type mat(matSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type XtX(XtXSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type Y(YSEXP);
    Rcpp::traits::input_parameter< int >::type n_threads(n_threadsSEXP);
    als_implicit(mat, X, XtX, Y, n_threads);
    return R_NilValue;
END_RCPP
}