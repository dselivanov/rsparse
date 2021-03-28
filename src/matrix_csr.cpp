#include <algorithm>
#include <unordered_map>
#include "rsparse.h"
#include <Rcpp.h>
#include <Rcpp/unwindProtect.h>
// [[Rcpp::plugins(unwindProtect)]]

// [[Rcpp::export(rng = false)]]
bool check_is_seq(Rcpp::IntegerVector indices) {
  if (indices.size() < 2) return true;
  int n_els = indices.size();
  if ((indices[n_els - 1] - indices[0]) != n_els - 1) return false;
  for (int ix = 1; ix < n_els; ix++) {
    if (indices[ix] != indices[ix - 1] + 1) return false;
  }
  return true;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List copy_csr_rows(Rcpp::IntegerVector indptr, Rcpp::IntegerVector indices,
                         Rcpp::NumericVector values, Rcpp::IntegerVector rows_take) {
  size_t total_size = 0;
  for (const int row : rows_take) total_size += indptr[row + 1] - indptr[row];
  if (total_size == 0) {
    return Rcpp::List::create(Rcpp::_["indptr"] = Rcpp::IntegerVector(),
                              Rcpp::_["indices"] = Rcpp::IntegerVector(),
                              Rcpp::_["values"] = Rcpp::NumericVector());
  }
  Rcpp::IntegerVector new_indptr = Rcpp::IntegerVector(rows_take.size() + 1);
  Rcpp::IntegerVector new_indices = Rcpp::IntegerVector(total_size);
  Rcpp::NumericVector new_values = Rcpp::NumericVector(values.size()? total_size : 0);

  size_t n_copy;
  int row;
  int* ptr_indptr = indptr.begin();
  int* ptr_indices = indices.begin();
  double* prt_values = values.begin();
  int* ptr_new_indptr = new_indptr.begin();
  int* ptr_new_indices = new_indices.begin();
  double* ptr_new_values = new_values.begin();
  const bool has_values = values.size() > 0;

  size_t curr = 0;
  for (int ix = 0; ix < (int)rows_take.size(); ix++) {
    row = rows_take[ix];
    n_copy = ptr_indptr[row + 1] - ptr_indptr[row];
    ptr_new_indptr[ix + 1] = ptr_new_indptr[ix] + n_copy;
    if (n_copy) {
      std::copy(ptr_indices + ptr_indptr[row], ptr_indices + ptr_indptr[row + 1],
                ptr_new_indices + curr);
      if (has_values)
        std::copy(prt_values + ptr_indptr[row], prt_values + ptr_indptr[row + 1],
                  ptr_new_values + curr);
    }
    curr += n_copy;
  }
  return Rcpp::List::create(Rcpp::_["indptr"] = new_indptr,
                            Rcpp::_["indices"] = new_indices,
                            Rcpp::_["values"] = new_values);
}

// [[Rcpp::export(rng = false)]]
Rcpp::List copy_csr_rows_col_seq(Rcpp::IntegerVector indptr, Rcpp::IntegerVector indices,
                                 Rcpp::NumericVector values,
                                 Rcpp::IntegerVector rows_take,
                                 Rcpp::IntegerVector cols_take) {
  int min_col = *std::min_element(cols_take.begin(), cols_take.end());
  int max_col = *std::max_element(cols_take.begin(), cols_take.end());
  Rcpp::IntegerVector new_indptr(rows_take.size() + 1);

  int* ptr_indptr = indptr.begin();
  int* ptr_indices = indices.begin();
  double* ptr_values = values.begin();
  int* ptr_new_indptr = new_indptr.begin();
  const bool has_values = values.size() > 0;

  size_t total_size = 0;
  for (int row = 0; row < (int)rows_take.size(); row++) {
    for (int ix = ptr_indptr[rows_take[row]]; ix < ptr_indptr[rows_take[row] + 1]; ix++) {
      total_size += (ptr_indices[ix] >= min_col) && (ptr_indices[ix] <= max_col);
    }
    ptr_new_indptr[row + 1] = total_size;
  }

  if (total_size == 0) {
    return Rcpp::List::create(Rcpp::_["indptr"] = new_indptr,
                              Rcpp::_["indices"] = Rcpp::IntegerVector(),
                              Rcpp::_["values"] = Rcpp::NumericVector());
  }

  Rcpp::IntegerVector new_indices = Rcpp::IntegerVector(total_size);
  Rcpp::NumericVector new_values = Rcpp::NumericVector(has_values? total_size : 0);
  int* ptr_new_indices = new_indices.begin();
  double* ptr_new_values = new_values.begin();

  int curr = 0;
  for (int row = 0; row < (int)rows_take.size(); row++) {
    for (int ix = ptr_indptr[rows_take[row]]; ix < ptr_indptr[rows_take[row] + 1]; ix++) {
      if ((ptr_indices[ix] >= min_col) && (ptr_indices[ix] <= max_col)) {
        ptr_new_indices[curr] = ptr_indices[ix] - min_col;
        if (has_values)
          ptr_new_values[curr] = ptr_values[ix];
        curr++;
      }
    }
  }
  return Rcpp::List::create(Rcpp::_["indptr"] = new_indptr,
                            Rcpp::_["indices"] = new_indices,
                            Rcpp::_["values"] = new_values);
}

struct VectorConstructorArgs {
  bool as_integer = false;
  bool from_cpp_vec = false;
  size_t size = 0;
  std::vector<int> *int_vec_from = NULL;
  std::vector<double> *num_vec_from = NULL;
};

SEXP SafeRcppVector(void *args_)
{
  VectorConstructorArgs *args = (VectorConstructorArgs*)args_;
  if (args->as_integer) {
    if (args->from_cpp_vec) {
      return Rcpp::IntegerVector(args->int_vec_from->begin(), args->int_vec_from->end());
    }

    else {
      return Rcpp::IntegerVector(args->size);
    }
  }

  else {
    if (args->from_cpp_vec) {
      return Rcpp::NumericVector(args->num_vec_from->begin(), args->num_vec_from->end());
    }

    else {
      return Rcpp::NumericVector(args->size);
    }
  }
}

// [[Rcpp::export(rng = false)]]
Rcpp::List copy_csr_arbitrary(Rcpp::IntegerVector indptr, Rcpp::IntegerVector indices,
                              Rcpp::NumericVector values, Rcpp::IntegerVector rows_take,
                              Rcpp::IntegerVector cols_take) {
  std::unordered_map<int, int> new_mapping;
  for (int col = 0; col < (int)cols_take.size(); col++) new_mapping[cols_take[col]] = col;
  std::unordered_map<int, int> n_repeats;
  for (auto el : cols_take) n_repeats[el]++;
  bool has_duplicates = false;
  for (auto& el : n_repeats) {
    if (el.second > 1) {
      has_duplicates = true;
      break;
    }
  }
  std::unordered_map<int, std::vector<int>> indices_rep;
  if (has_duplicates) {
    for (int col = 0; col < (int)cols_take.size(); col++) {
      if (n_repeats[cols_take[col]] > 1) {
        indices_rep[cols_take[col]].push_back(col);
      }
    }
  }

  bool cols_are_sorted = true;
  for (int ix = 1; ix < (int)cols_take.size(); ix++) {
    if (cols_take[ix] < cols_take[ix - 1]) {
      cols_are_sorted = false;
      break;
    }
  }

  Rcpp::IntegerVector new_indptr;
  VectorConstructorArgs args;
  args.as_integer = true; args.from_cpp_vec = false; args.size = rows_take.size() + 1;
  new_indptr = Rcpp::unwindProtect(SafeRcppVector, (void*)&args);
  const char *oom_err_msg = "Could not allocate sufficient memory.\n";
  if (!new_indptr.size())
    Rcpp::stop(oom_err_msg);
  std::vector<int> new_indices;
  std::vector<double> new_values;

  std::vector<int> argsort_cols;
  std::vector<int> temp_int;
  std::vector<double> temp_double;

  const bool has_values = values.size() > 0;

  int size_this = 0;
  int row = 0;
  for (int row_ix = 0; row_ix < (int)rows_take.size(); row_ix++) {
    row = rows_take[row_ix];
    for (int ix = indptr[row]; ix < indptr[row + 1]; ix++) {
      auto match = new_mapping.find(indices[ix]);
      if (match != new_mapping.end()) {
        if (has_duplicates && n_repeats[indices[ix]] > 1) {
          for (const auto& el : indices_rep[indices[ix]]) {
            new_indices.push_back(el);
            if (has_values)
              new_values.push_back(values[ix]);
          }
        } else {
          new_indices.push_back(match->second);
          if (has_values)
            new_values.push_back(values[ix]);
        }
      }
    }
    new_indptr[row_ix + 1] = new_indices.size();
    if (!cols_are_sorted && new_indptr[row_ix + 1] > new_indptr[row_ix]) {
      size_this = new_indptr[row_ix + 1] - new_indptr[row_ix];
      if (argsort_cols.size() < (size_t)size_this) {
        argsort_cols.resize(size_this);
        temp_int.resize(size_this);
        temp_double.resize(size_this);
      }
      std::iota(argsort_cols.begin(), argsort_cols.begin() + size_this,
                new_indptr[row_ix]);
      std::sort(argsort_cols.begin(), argsort_cols.begin() + size_this,
                [&new_indices](const int a, const int b) {
                  return new_indices[a] < new_indices[b];
                });
      for (int col = 0; col < size_this; col++) {
        temp_int[col] = new_indices[argsort_cols[col]];
        if (has_values)
          temp_double[col] = new_values[argsort_cols[col]];
      }
      std::copy(temp_int.begin(), temp_int.begin() + size_this,
                new_indices.begin() + new_indptr[row_ix]);
      if (has_values)
        std::copy(temp_double.begin(), temp_double.begin() + size_this,
                  new_values.begin() + new_indptr[row_ix]);
    }
  }

  Rcpp::List out;
  out["indptr"] = new_indptr;
  args.as_integer = true; args.from_cpp_vec = true; args.int_vec_from = &new_indices;
  out["indices"] = Rcpp::unwindProtect(SafeRcppVector, (void*)&args);
  if (Rf_xlength(out["indices"]) != new_indices.size())
    Rcpp::stop(oom_err_msg);
  new_indices.clear();
  if (values.size()) {
    args.as_integer = false; args.from_cpp_vec = true; args.num_vec_from = &new_values;
    out["values"] = Rcpp::unwindProtect(SafeRcppVector, (void*)&args);
    if (Rf_xlength(out["values"]) != new_values.size())
      Rcpp::stop(oom_err_msg);
  }
  return out;
}

// [[Rcpp::export(rng = false)]]
Rcpp::IntegerVector repeat_indices_n_times(Rcpp::IntegerVector indices, Rcpp::IntegerVector remainder, int ix_length, int desired_length)
{
  int full_repeats = desired_length / ix_length;
  auto n_indices = indices.size();
  Rcpp::IntegerVector out(n_indices*full_repeats + remainder.size());
  for (int repetition = 0; repetition < full_repeats; repetition++) {
    #pragma omp simd
    for (int ix = 0; ix < n_indices; ix++)
      out[ix + n_indices*repetition] = indices[ix] + ix_length*repetition;
  }
  #pragma omp simd
  for (int ix = 0; ix < remainder.size(); ix++)
    out[ix + n_indices*full_repeats] = remainder[ix] + ix_length*full_repeats;
  return out;
}

// [[Rcpp::export(rng = false)]]
Rcpp::IntegerVector concat_indptr2(Rcpp::IntegerVector ptr1, Rcpp::IntegerVector ptr2)
{
  Rcpp::IntegerVector out(ptr1.size() + ptr2.size() - 1);
  std::copy(INTEGER(ptr1), INTEGER(ptr1) + ptr1.size(), INTEGER(out));
  size_t st_second = ptr1.size();
  int offset = ptr1[ptr1.size()-1];
  #pragma omp simd
  for (size_t row = 1; row < ptr2.size(); row++)
    out[st_second + row - 1] = offset + ptr2[row];
  return out;
}

enum RbindedType {dgRMatrix, lgRMatrix, ngRMatrix};

// [[Rcpp::export(rng = false)]]
Rcpp::S4 concat_csr_batch(Rcpp::ListOf<Rcpp::S4> objects, Rcpp::S4 out)
{
  size_t n_inputs = objects.size();
  RbindedType otype;
  if (out.inherits("ngRMatrix")) {
    otype = ngRMatrix;
  } else if (out.inherits("lgRMatrix")) {
    otype = lgRMatrix;
  } else {
    otype = dgRMatrix;
  }
  int *indptr_out = INTEGER(out.slot("p"));
  int *indices_out = INTEGER(out.slot("j"));
  double *values_out = (otype == dgRMatrix)? REAL(out.slot("x")) : nullptr;
  int *values_out_bool = (otype == lgRMatrix)? LOGICAL(out.slot("x")) : nullptr;

  int nrows_add, nnz_add;
  int *indptr_obj, *indices_obj;
  double *xvals_obj = nullptr;
  int *lvals_obj = nullptr;
  int *ivals_obj = nullptr;

  indptr_out[0] = 0;
  int curr_pos = 0;
  int curr_row = 0;

  for (size_t ix = 0; ix < n_inputs; ix++) {

    if (objects[ix].hasSlot("j")) {
      
      indptr_obj = INTEGER(objects[ix].slot("p"));
      indices_obj = INTEGER(objects[ix].slot("j"));
      nnz_add = Rf_xlength(objects[ix].slot("j"));
      xvals_obj = objects[ix].inherits("dgRMatrix")? REAL(objects[ix].slot("x")) : nullptr;
      lvals_obj = objects[ix].inherits("lgRMatrix")? LOGICAL(objects[ix].slot("x")) : nullptr;
      nrows_add = INTEGER(objects[ix].slot("Dim"))[0];

      /* indptr */
      for (int row = 0; row < nrows_add; row++)
        indptr_out[row + curr_row + 1] = indptr_out[curr_row] + indptr_obj[row+1];
      curr_row += nrows_add;
      /* indices */
      std::copy(indices_obj, indices_obj + nnz_add, indices_out + curr_pos);
      /* values, if applicable */
      if (otype == dgRMatrix) {
        if (xvals_obj != nullptr)
          std::copy(xvals_obj, xvals_obj + nnz_add, values_out + curr_pos);
        else if (lvals_obj != nullptr)
          #pragma omp simd
          for (int el = 0; el < nnz_add; el++)
            values_out[el + curr_pos] = (lvals_obj[el] == NA_LOGICAL)? (NA_REAL) : lvals_obj[el];
        else
          std::fill(values_out + curr_pos, values_out + curr_pos + nnz_add, 1.);
      }
      else if (otype == lgRMatrix) {
        if (lvals_obj != nullptr)
          std::copy(lvals_obj, lvals_obj + nnz_add, values_out_bool + curr_pos);
        else if (xvals_obj != nullptr)
          #pragma omp simd
          for (int el = 0; el < nnz_add; el++)
            values_out_bool[el + curr_pos] = ISNAN(xvals_obj[el])? NA_LOGICAL : (bool)xvals_obj[el];
        else
          std::fill(values_out_bool + curr_pos, values_out_bool + curr_pos + nnz_add, (int)true);
      }
    }

    else {
      
      indices_obj = INTEGER(objects[ix].slot("i"));
      nnz_add = Rf_xlength(objects[ix].slot("i"));
      indptr_out[curr_row + 1] = indptr_out[curr_row] + nnz_add;
      curr_row++;
      #pragma omp simd
      for (int el = 0; el < nnz_add; el++)
        indices_out[el + curr_pos] = indices_obj[el] - 1;

      if (otype == dgRMatrix) {
        
        if (objects[ix].inherits("dsparseVector")) {
          xvals_obj = REAL(objects[ix].slot("x"));
          std::copy(xvals_obj, xvals_obj + nnz_add, values_out + curr_pos);
        } else if (objects[ix].inherits("isparseVector")) {
          ivals_obj = INTEGER(objects[ix].slot("x"));
          #pragma omp simd
          for (int el = 0; el < nnz_add; el++)
            values_out[el + curr_pos] = (ivals_obj[el] == NA_INTEGER)? NA_REAL : ivals_obj[el];
        } else if (objects[ix].inherits("lsparseVector")) {
          lvals_obj = LOGICAL(objects[ix].slot("x"));
          #pragma omp simd
          for (int el = 0; el < nnz_add; el++)
            values_out[el + curr_pos] = (lvals_obj[el] == NA_LOGICAL)? NA_REAL : (bool)lvals_obj[el];
        } else if (objects[ix].inherits("nsparseVector")) {
            std::fill(values_out + curr_pos, values_out + curr_pos + nnz_add, 1.);
        } else {
          char errmsg[100];
          std::snprintf(errmsg, 99, "Invalid vector type in argument %d.\n", (int)ix);
          Rcpp::stop(errmsg);
        }

      } else if (otype == lgRMatrix) {

        if (objects[ix].inherits("dsparseVector")) {
          xvals_obj = REAL(objects[ix].slot("x"));
          #pragma omp simd
          for (int el = 0; el < nnz_add; el++)
            values_out_bool[el + curr_pos] = ISNAN(xvals_obj[el])? NA_LOGICAL : (bool)xvals_obj[el];
        } else if (objects[ix].inherits("isparseVector")) {
          ivals_obj = INTEGER(objects[ix].slot("x"));
          #pragma omp simd
          for (int el = 0; el < nnz_add; el++)
            values_out_bool[el + curr_pos] = (ivals_obj[el] == NA_INTEGER)? NA_LOGICAL : (bool)ivals_obj[el];
        } else if (objects[ix].inherits("lsparseVector")) {
          lvals_obj = LOGICAL(objects[ix].slot("x"));
          std::copy(lvals_obj, lvals_obj + nnz_add, values_out_bool + curr_pos);
        } else if (objects[ix].inherits("nsparseVector")) {
          std::fill(values_out_bool + curr_pos, values_out_bool + curr_pos + nnz_add, (int)true);
        } else {
          char errmsg[100];
          std::snprintf(errmsg, 99, "Invalid vector type in argument %d.\n", (int)ix);
          Rcpp::stop(errmsg);
        }

      }
    }

    curr_pos += nnz_add;
  }

  return out;
}
