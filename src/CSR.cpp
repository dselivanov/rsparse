#include "rsparse.h"
#include <algorithm>
#include <unordered_map>

// [[Rcpp::export]]
bool check_is_seq(Rcpp::IntegerVector indices)
{
  if (indices.size() < 2)
    return false;
  int n_els = indices.size();
  if ((indices[n_els-1] - indices[0]) != n_els - 1)
    return false;
  for (int ix = 1; ix < n_els; ix++) {
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

// [[Rcpp::export]]
Rcpp::List copy_csr_rows_col_seq(Rcpp::IntegerVector indptr,
                                 Rcpp::IntegerVector indices,
                                 Rcpp::NumericVector values,
                                 Rcpp::IntegerVector rows_take,
                                 Rcpp::IntegerVector cols_take)
{
  int min_col = *std::min_element(cols_take.begin(), cols_take.end());
  int max_col = *std::max_element(cols_take.begin(), cols_take.end());
  Rcpp::IntegerVector new_indptr(rows_take.size() + 1);

  int *ptr_indptr = indptr.begin();
  int *ptr_indices = indices.begin();
  double *ptr_values = values.begin();
  int *ptr_new_indptr = new_indptr.begin();

  size_t total_size = 0;
  for (size_t row = 0; row < rows_take.size(); row++) {
    for (int ix = ptr_indptr[rows_take[row]]; ix < ptr_indptr[rows_take[row]+1]; ix++) {
      total_size += (ptr_indices[ix] >= min_col) && (ptr_indices[ix] <= max_col);
    }
    ptr_new_indptr[row+1] = total_size;
  }

  if (total_size == 0) {
    return Rcpp::List::create(
      Rcpp::_["indptr"] = new_indptr,
      Rcpp::_["indices"] = Rcpp::IntegerVector(),
      Rcpp::_["values"] = Rcpp::NumericVector()
    );
  }

  Rcpp::IntegerVector new_indices = Rcpp::IntegerVector(total_size);
  Rcpp::NumericVector new_values = Rcpp::NumericVector(total_size);
  int *ptr_new_indices = new_indices.begin();
  double *ptr_new_values = new_values.begin();

  int curr = 0;
  for (size_t row = 0; row < rows_take.size(); row++) {
    for (int ix = ptr_indptr[rows_take[row]]; ix < ptr_indptr[rows_take[row]+1]; ix++) {
      if ((ptr_indices[ix] >= min_col) && (ptr_indices[ix] <= max_col)) {
        ptr_new_indices[curr] = ptr_indices[ix] - min_col;
        ptr_new_values[curr] = ptr_values[ix];
        curr++;
      }
    }
  }
  return Rcpp::List::create(
    Rcpp::_["indptr"] = new_indptr,
    Rcpp::_["indices"] = new_indices,
    Rcpp::_["values"] = new_values
  );
}

// [[Rcpp::export]]
Rcpp::List copy_csr_arbitrary(Rcpp::IntegerVector indptr,
                              Rcpp::IntegerVector indices,
                              Rcpp::NumericVector values,
                              Rcpp::IntegerVector rows_take,
                              Rcpp::IntegerVector cols_take)
{
  size_t total_size = 0;
  std::unordered_map<int, int> new_mapping;
  for (int col = 0; col < (int)cols_take.size(); col++)
    new_mapping[cols_take[col]] = col;
  std::unordered_map<int, int> n_repeats;
  for (auto el : cols_take)
    n_repeats[el]++;
  bool has_duplicates = false;
  for (auto &el : n_repeats) {
    if (el.second > 1) {
      has_duplicates = true;
      break;
    }
  }

  bool cols_are_sorted = !has_duplicates;
  if (!has_duplicates) {
    for (size_t ix = 1; ix < cols_take.size(); ix++) {
      if (cols_take[ix] < cols_take[ix-1]) {
        cols_are_sorted = false;
        break;
      }
    }
  }

  Rcpp::IntegerVector new_indptr = Rcpp::IntegerVector(rows_take.size() + 1);
  std::vector<int> new_indices;
  std::vector<double> new_values;

  std::vector<int> argsort_cols;
  std::vector<int> temp_int;
  std::vector<double> temp_double;

  int size_this = 0;
  int row = 0;
  int rep = 0;
  for (size_t row_ix = 0; row_ix < rows_take.size(); row_ix++) {
    row = rows_take[row_ix];
    for (int ix = indptr[row]; ix < indptr[row+1]; ix++) {
      auto match = new_mapping.find(indices[ix]);
      if (match != new_mapping.end()) {
        new_indices.push_back(match->second);
        new_values.push_back(values[ix]);
        if (has_duplicates && n_repeats[indices[ix]] > 1) {
          rep = n_repeats[indices[ix]];
          for (int r = 1; r < rep; r++) {
            new_indices.push_back(new_indices.back()-1);
            new_values.push_back(values[ix]);
          }
        }
      }
    }
    new_indptr[row_ix+1] = new_indices.size();
    if (!cols_are_sorted && new_indptr[row_ix+1] > new_indptr[row_ix]) {
      size_this = new_indptr[row_ix+1] - new_indptr[row_ix];
      if (argsort_cols.size() < (size_t)size_this) {
        argsort_cols.resize(size_this);
        temp_int.resize(size_this);
        temp_double.resize(size_this);
      }
      std::iota(argsort_cols.begin(), argsort_cols.end(), new_indptr[row_ix]);
      std::sort(argsort_cols.begin(), argsort_cols.end(),
                [&new_indices](const int a, const int b){return new_indices[a] < new_indices[b];});
      for (int col = 0; col < size_this; col++) {
        temp_int[col] = new_indices[argsort_cols[col]];
        temp_double[col] = new_values[argsort_cols[col]];
      }
      std::copy(temp_int.begin(),
                temp_int.begin() + size_this,
                new_indices.begin() + new_indptr[row_ix]);
      std::copy(temp_double.begin(),
                temp_double.begin() + size_this,
                new_values.begin() + new_indptr[row_ix]);
    }
  }
  return Rcpp::List::create(
    Rcpp::_["indptr"] = new_indptr,
    Rcpp::_["indices"] = Rcpp::IntegerVector(new_indices.begin(), new_indices.end()),
    Rcpp::_["values"] = Rcpp::NumericVector(new_values.begin(), new_values.end())
  );
}

