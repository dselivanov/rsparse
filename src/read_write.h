#if (defined(_WIN32) || defined(_WIN64)) && (defined(__GNUG__) || defined(__GNUC__)) && (SIZE_MAX > UINT32_MAX) && !defined(_FILE_OFFSET_BITS)
#  define _FILE_OFFSET_BITS 64 /* https://stackoverflow.com/questions/16696297/ftell-at-a-position-past-2gb */
#endif

/* Aliasing for compiler optimizations */
#if defined(__GNUG__) || defined(__GNUC__) || defined(__clang__) || defined(__INTEL_COMPILER)
  #define restrict __restrict
#else
  #define restrict
#endif

#include <Rcpp.h>
#include <Rcpp/unwindProtect.h>
#include <Rinternals.h>
// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::plugins(unwindProtect)]]

#include <cinttypes>
#include <stdint.h>
#include <stdio.h>
#include <errno.h>
#include <stddef.h>
#include <limits.h>
#include <cmath>
#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <regex>
#include <numeric>
#include <algorithm>
#include <unordered_map>

#define missing_qid NA_INTEGER
#define throw_errno(num) Rcpp::Rcerr << "Error " << num << ": " << strerror(num) << std::endl

extern "C" {
  FILE *RC_fopen(const SEXP fn, const char *mode, const Rboolean expand);
}

static inline SEXP convert_IntVecToRcpp(void *data)
{
  return Rcpp::IntegerVector(((std::vector<int>*)data)->begin(),
                             ((std::vector<int>*)data)->end());
}

static inline SEXP convert_NumVecToRcpp(void *data)
{
  return Rcpp::NumericVector(((std::vector<double>*)data)->begin(),
                             ((std::vector<double>*)data)->end());
}

static inline SEXP convert_StringStreamToRcpp(void *data)
{
  return Rcpp::CharacterVector(((std::stringstream*)data)->str());
}



/* Prototypes of functions to take from 'matrix_csr.cpp' */
bool check_is_sorted(int* vec, size_t n);
void sort_sparse_indices
(
  int *indptr,
  int *indices,
  double *values,
  size_t nrows,
  bool has_values
);
void sort_sparse_indices
(
  std::vector<int> &indptr,
  std::vector<int> &indices,
  std::vector<double> &values
);
void sort_sparse_indices
(
  int *indptr,
  int *indices,
  size_t nrows
);
void sort_sparse_indices
(
  int *indptr,
  int *indices,
  double *values,
  size_t nrows
);
