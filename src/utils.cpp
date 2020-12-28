#include "rsparse.h"
#include <time.h>

// [[Rcpp::export]]
Rcpp::NumericVector cpp_make_sparse_approximation(const Rcpp::S4 &mat_template,
                                            const arma::mat& X,
                                            const arma::mat& Y,
                                            int sparse_matrix_type,
                                            unsigned n_threads) {
  Rcpp::IntegerVector rp = mat_template.slot("p");
  int* p = rp.begin();
  Rcpp::IntegerVector rj;
  if(sparse_matrix_type == CSR) {
    rj = mat_template.slot("j");
  } else if(sparse_matrix_type == CSC) {
    rj = mat_template.slot("i");
  } else
    Rcpp::stop("make_sparse_approximation_csr doesn't know sparse matrix type. Should be CSC=1 or CSR=2");

  uint32_t* j = (uint32_t *)rj.begin();
  Rcpp::IntegerVector dim = mat_template.slot("Dim");

  size_t nr = dim[0];
  size_t nc = dim[1];
  uint32_t N;
  if(sparse_matrix_type == CSR)
    N = nr;
  else
    N = nc;

  Rcpp::NumericVector approximated_values(rj.length());

  double *ptr_approximated_values = approximated_values.begin();
  #ifdef _OPENMP
  #pragma omp parallel for num_threads(n_threads) schedule(dynamic, GRAIN_SIZE)
  #endif
  for(uint32_t i = 0; i < N; i++) {
    int p1 = p[i];
    int p2 = p[i + 1];
    arma::rowvec xc;
    if(sparse_matrix_type == CSR)
      xc = X.col(i).t();
    else
      xc = Y.col(i).t();

    for(int pp = p1; pp < p2; pp++) {
      uint64_t ind = (size_t)j[pp];
      if(sparse_matrix_type == CSR)
        ptr_approximated_values[pp] = as_scalar(xc * Y.col(ind));
      else
        ptr_approximated_values[pp] = as_scalar(xc * X.col(ind));
    }

  }
  return(approximated_values);
}

dMappedCSR extract_mapped_csr(Rcpp::S4 input) {
  Rcpp::IntegerVector dim = input.slot("Dim");
  Rcpp::NumericVector values = input.slot("x");
  arma::uword nrows = dim[0];
  arma::uword ncols = dim[1];
  Rcpp::IntegerVector rj = input.slot("j");
  Rcpp::IntegerVector rp = input.slot("p");
  return dMappedCSR(nrows, ncols, values.length(), (arma::uword *)rj.begin(), (arma::uword *)rp.begin(), (double *)values.begin());
}

dMappedCSC extract_mapped_csc(Rcpp::S4 input) {
  Rcpp::IntegerVector dim = input.slot("Dim");
  Rcpp::NumericVector values = input.slot("x");
  arma::uword nrows = dim[0];
  arma::uword ncols = dim[1];
  Rcpp::IntegerVector row_indices = input.slot("i");
  Rcpp::IntegerVector col_ptrs = input.slot("p");
  return dMappedCSC(nrows, ncols, values.length(), (arma::uword *)row_indices.begin(), (arma::uword *)col_ptrs.begin(), (double *)values.begin());
}

// [[Rcpp::export]]
Rcpp::IntegerVector convert_indptr_to_rows(Rcpp::IntegerVector indptr, int n)
{
  /* Note: the output will have numeration starting at 1  */
  Rcpp::IntegerVector res(n, 0);
  if (n == 0) return res;
  int curr = 0;
  for (size_t ix = 1; ix < indptr.size(); ix++) {
    if (indptr[ix] > indptr[ix-1])
      res[curr++] = ix;
  }
  return res;
}

// returns number of available threads
// omp_get_num_threads() for some reason doesn't work on all systems
// check following link
// http://stackoverflow.com/questions/11071116/i-got-omp-get-num-threads-always-return-1-in-gcc-works-in-icc
int omp_thread_count() {
  int n = 0;
  #ifdef _OPENMP
  #pragma omp parallel reduction(+:n)
  #endif
  n += 1;
  return n;
}


bool is_master() {
  #ifdef _OPENMP
  return omp_get_thread_num() == 0;
  #else
  return true;
  #endif
}


// https://stackoverflow.com/a/10467633/1069256
// Get current date/time, format is YYYY-MM-DD HH:mm:ss
const std::string currentDateTime() {
  time_t     now = time(0);
  struct tm  tstruct;
  char       buf[80];
  tstruct = *localtime(&now);
  // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
  // for more information about date/time format
  strftime(buf, sizeof(buf), "%Y-%m-%d %X", &tstruct);

  return buf;
}

arma::fmat extract_float_matrix(Rcpp::S4 x) {
  Rcpp::IntegerMatrix x_data = x.slot("Data");
  if (!x_data.size())
    return arma::fmat();
  float *ptr = reinterpret_cast<float *>(&x_data[0]);
  arma::fmat x_mapped = arma::fmat(ptr, x_data.nrow(), x_data.ncol(), false, true);
  return (x_mapped);
}

arma::fvec extract_float_vector(Rcpp::S4 x) {
  Rcpp::IntegerVector x_data = x.slot("Data");
  if (!x_data.size())
    return arma::fvec();
  float *ptr = reinterpret_cast<float *>(&x_data[0]);
  arma::fvec x_mapped = arma::fvec(ptr, x_data.length(), false, true);
  return (x_mapped);
}

// [[Rcpp::export]]
SEXP large_rand_matrix(SEXP nrow, SEXP ncol)
{
  int nrow_int = Rf_asInteger(nrow);
  int ncol_int = Rf_asInteger(ncol);
  R_xlen_t tot_size = (R_xlen_t)nrow_int * (R_xlen_t)ncol_int;
  if (tot_size <= 0 || nrow_int <= 0 || ncol_int <= 0)
    Rf_error("Factors dimensions exceed R limits.");
  SEXP vec = PROTECT(Rf_allocMatrix(REALSXP, nrow_int, ncol_int));
  double *ptr_vec = REAL(vec);
  for (R_xlen_t ix = 0; ix < tot_size; ix++)
    ptr_vec[ix] = norm_rand();
  for (R_xlen_t ix = 0; ix < tot_size; ix++)
    ptr_vec[ix] /= 100.;
  UNPROTECT(1);
  return vec;
}

// [[Rcpp::export]]
SEXP deep_copy(SEXP x) {
  SEXP out = PROTECT(Rf_allocVector(REALSXP, Rf_xlength(x)));
  if (Rf_xlength(x))
    memcpy(REAL(out), REAL(x), (size_t)Rf_xlength(x)*sizeof(double));
  UNPROTECT(1);
  return out;
}
