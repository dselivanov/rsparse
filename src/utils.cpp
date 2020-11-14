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

// [[Rcpp::export]]
bool check_is_seq(Rcpp::IntegerVector indices)
{
  if (indices.size() < 2)
    return false;
  int n_els = indices.size();
  if ((indices[n_els-1] - indices[0]) != n_els - 1)return false;
  for (size_t ix = 1; ix < n_els; ix++) {
    if (indices[ix] != indices[ix-1] + 1)
      return false;
  }
  return true;
}

// [[Rcpp::export]]
Rcpp::List copy_csr_rows(Rcpp::IntegerVector indptr,
                         Rcpp::IntegerVector indices,
                         Rcpp::NumericVector values,
                         Rcpp::IntegerVector rows_take)
{
  size_t total_size = 0;
  for (const int row : rows_take)
    total_size += indptr[row+1] - indptr[row];
  if (total_size == 0) {
    return Rcpp::List::create(
      Rcpp::_["indptr"] = Rcpp::IntegerVector(),
      Rcpp::_["indices"] = Rcpp::IntegerVector(),
      Rcpp::_["values"] = Rcpp::NumericVector()
    );
  }
  Rcpp::IntegerVector new_indptr = Rcpp::IntegerVector(rows_take.size() + 1);
  Rcpp::IntegerVector new_indices = Rcpp::IntegerVector(total_size);
  Rcpp::NumericVector new_values = Rcpp::NumericVector(total_size);

  size_t n_copy;
  int row;
  int *ptr_indptr = indptr.begin();
  int *ptr_indices = indices.begin();
  double *prt_values = values.begin();
  int *ptr_new_indptr = new_indptr.begin();
  int *ptr_new_indices = new_indices.begin();
  double *ptr_new_values = new_values.begin();

  size_t curr = 0;
  for (size_t ix = 0; ix < rows_take.size(); ix++) {
    row = rows_take[ix];
    n_copy = ptr_indptr[row+1] - ptr_indptr[row];
    ptr_new_indptr[ix+1] = ptr_new_indptr[ix] + n_copy;
    if (n_copy) {
      std::copy(ptr_indices + ptr_indptr[row],
                ptr_indices + ptr_indptr[row + 1],
                ptr_new_indices + curr);
      std::copy(prt_values + ptr_indptr[row],
                prt_values + ptr_indptr[row + 1],
                ptr_new_values + curr);
    }
    curr += n_copy;
  }
  return Rcpp::List::create(
    Rcpp::_["indptr"] = new_indptr,
    Rcpp::_["indices"] = new_indices,
    Rcpp::_["values"] = new_values
  );
}

