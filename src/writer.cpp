#include "read_write.h"
#define above_min_decimals(x) (x && std::abs(x) >= thr_decimals)

bool write_multi_label_cpp
(
  std::ostream &output_file,
  int *restrict indptr,
  int *restrict indices,
  double *restrict values,
  int *restrict indptr_lab,
  int *restrict indices_lab,
  int *restrict qid,
  const bool has_qid,
  const int nrows,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places
)
{
  if (output_file.fail())
  {
    Rcpp::Rcerr << "Error: invalid output_file." << std::endl;
    return false;
  }

  if (decimal_places < 0)
  {
    Rcpp::Rcerr << "Error: 'decimal_places' cannot be negative." << std::endl;
    return false;
  }

  if (nrows == 0 && !add_header)
     return true;

  bool succeded = true;
  double thr_decimals = std::pow(10., (double)(-decimal_places)) / 2.;
  size_t n_this, ix1, ix2;

  /* https://stackoverflow.com/questions/8554441/how-to-output-with-3-digits-after-the-decimal-point-with-c-stream */
  std::ios_base::fmtflags oldflags = output_file.flags();
  std::streamsize oldprecision = output_file.precision();
  output_file << std::fixed << std::setprecision(decimal_places);

  if (add_header)
  {
    output_file << nrows << ' ' << ncols << ' ' << nclasses << '\n';
    if (output_file.bad()) goto terminate_badly;
  }

  if (nrows == 0)
    goto terminate_func;

  if (sort_indices)
  {
    sort_sparse_indices(indptr, indices, values, nrows);
    sort_sparse_indices(indptr_lab, indices_lab, nrows);
  }

  for (size_t row = 0; row < nrows; row++)
  {
    ix1 = indptr_lab[row];
    ix2 = indptr_lab[row+1];
    n_this = ix2 - ix1;
    if (n_this == 1)
    {
      output_file << (indices_lab[ix1] + text_is_base1);
      if (output_file.bad()) goto terminate_badly;
    }

    else if (n_this > 1)
    {
      for (size_t ix = ix1; ix < ix2-1; ix++) {
        output_file << (indices_lab[ix] + text_is_base1) << ',';
        if (output_file.bad()) goto terminate_badly;
      }
      output_file << (indices_lab[ix2-1] + text_is_base1);
      if (output_file.bad()) goto terminate_badly;
    }

    output_file << ' ';
    if (output_file.bad()) goto terminate_badly;

    if (has_qid && qid[row] != missing_qid)
    {
      output_file << "qid:" << qid[row] << ' ';
      if (output_file.bad()) goto terminate_badly;
    }

    ix1 = indptr[row];
    ix2 = indptr[row+1];
    n_this = ix2 - ix1;
    if (n_this == 1)
    {
      if (!ignore_zero_valued || above_min_decimals(values[ix1]))
      {
        output_file << (indices[ix1] + text_is_base1) << ':' << values[ix1];
        if (output_file.bad()) goto terminate_badly;
      }
    }

    else if (n_this > 1)
    {
      for (size_t ix = ix1; ix < ix2-1; ix++)
      {
        if (!ignore_zero_valued || above_min_decimals(values[ix]))
        {
          output_file << (indices[ix] + text_is_base1) << ':' << values[ix] << ' ';
          if (output_file.bad()) goto terminate_badly;
        }
      }
      if (!ignore_zero_valued || above_min_decimals(values[ix2-1]))
      {
        output_file << (indices[ix2-1] + text_is_base1) << ':' << values[ix2-1];
        if (output_file.bad()) goto terminate_badly;
      }
    }

    output_file << '\n';
    if (output_file.bad()) goto terminate_badly;
  }

  terminate_func:
  output_file.flags(oldflags);
  output_file.precision(oldprecision);
  return succeded;

  terminate_badly:
  succeded = false;
  goto terminate_func;
}

bool write_multi_label_cpp
(
  FILE *output_file,
  int *restrict indptr,
  int *restrict indices,
  double *restrict values,
  int *restrict indptr_lab,
  int *restrict indices_lab,
  int *restrict qid,
  const bool has_qid,
  const int nrows,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places
)
{
  if (output_file == NULL)
  {
    Rcpp::Rcerr << "Error: invalid output_file." << std::endl;
    return false;
  }

  if (decimal_places < 0)
  {
    Rcpp::Rcerr << "Error: 'decimal_places' cannot be negative." << std::endl;
    return false;
  }

  int succeded;
  if (add_header)
  {
    succeded = fprintf(output_file, "%d %d %d\n",
               nrows, ncols, nclasses);
    if (succeded < 0) return false;
  }

  if (nrows == 0)
    return true;

  if (sort_indices)
  {
    sort_sparse_indices(indptr, indices, values, nrows);
    sort_sparse_indices(indptr_lab, indices_lab, nrows);
  }
  
  double thr_decimals = std::pow(10., (double)(-decimal_places)) / 2.;

  char format_specifier_spaced[23];
  char format_specifier[23];
  sprintf(format_specifier_spaced, "%%d:%%.%df ", decimal_places);
  sprintf(format_specifier, "%%d:%%.%df", decimal_places);


  size_t n_this, ix1, ix2;
  for (size_t row = 0; row < nrows; row++)
  {
    ix1 = indptr_lab[row];
    ix2 = indptr_lab[row+1];
    n_this = ix2 - ix1;

    if (n_this == 0)
    {
      succeded = fprintf(output_file, " ");
      if (succeded < 0) goto throw_err;
    }

    else if (n_this == 1)
    {
      succeded = fprintf(output_file, "%d ",
                 indices_lab[ix1] + text_is_base1);
      if (succeded < 0) goto throw_err;
    }

    else if (n_this > 1)
    {
      for (size_t ix = ix1; ix < ix2-1; ix++) {
        succeded = fprintf(output_file, "%d,",
                   indices_lab[ix] + text_is_base1);
        if (succeded < 0) goto throw_err;
      }

      succeded = fprintf(output_file, "%d ",
                 indices_lab[ix2-1] + text_is_base1);
      if (succeded < 0) goto throw_err;
    }

    if (has_qid && qid[row] != missing_qid)
    {
      succeded = fprintf(output_file, "qid:%d ", qid[row]);
      if (succeded < 0) goto throw_err;
    }

    ix1 = indptr[row];
    ix2 = indptr[row+1];
    n_this = ix2 - ix1;
    if (n_this == 1)
    {
      if (!ignore_zero_valued || above_min_decimals(values[ix1]))
      {
        succeded = fprintf(output_file, format_specifier,
                   indices[ix1] + text_is_base1, values[ix1]);
        if (succeded < 0) goto throw_err;
      }
    }

    else if (n_this > 1)
    {
      for (size_t ix = ix1; ix < ix2-1; ix++)
      {
        if (!ignore_zero_valued || above_min_decimals(values[ix]))
        {
          succeded = fprintf(output_file, format_specifier_spaced,
                     indices[ix] + text_is_base1, values[ix]);
          if (succeded < 0) goto throw_err;
        }
      }
      if (!ignore_zero_valued || above_min_decimals(values[ix2-1]))
      {
        succeded = fprintf(output_file, format_specifier,
                   indices[ix2-1] + text_is_base1, values[ix2-1]);
        if (succeded < 0) goto throw_err;
      }
    }

    succeded = fprintf(output_file, "\n");
    if (succeded < 0) goto throw_err;
  }

  return true;

  throw_err:
  {
    throw_errno(errno);
    return false;
  }
}

template <class label_t>
bool write_single_label_template
(
  std::ostream &output_file,
  int *restrict indptr,
  int *restrict indices,
  double *restrict values,
  label_t *restrict labels,
  int *restrict qid,
  const bool has_qid,
  const int nrows,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places
)
{
  if (output_file.fail())
  {
    Rcpp::Rcerr << "Error: invalid output_file." << std::endl;
    return false;
  }

  if (decimal_places < 0)
  {
    Rcpp::Rcerr << "Error: 'decimal_places' cannot be negative." << std::endl;
    return false;
  }

  if (nrows == 0 && !add_header)
     return true;

  bool succeded = true;
  double thr_decimals = std::pow(10., (double)(-decimal_places)) / 2.;
  size_t n_this, ix1, ix2;
  bool label_is_num = std::is_same<label_t, double>::value;

  /* https://stackoverflow.com/questions/8554441/how-to-output-with-3-digits-after-the-decimal-point-with-c-stream */
  std::ios_base::fmtflags oldflags = output_file.flags();
  std::streamsize oldprecision = output_file.precision();
  output_file << std::fixed << std::setprecision(decimal_places);

  if (add_header)
  {
    output_file << nrows << ' ' << ncols << ' ' << nclasses << '\n';
    if (output_file.bad()) goto terminate_badly;
  }

  if (nrows == 0)
    goto terminate_func;

  if (sort_indices)
  {
    sort_sparse_indices(indptr, indices, values, nrows);
  }

  for (size_t row = 0; row < nrows; row++)
  {
    if (label_is_num)
    {
      if (!ISNAN(labels[row]))
      {
        output_file << labels[row];
        if (output_file.bad()) goto terminate_badly;
      }
    }

    else
    {
      if (labels[row] != NA_INTEGER)
      {
        output_file << labels[row];
        if (output_file.bad()) goto terminate_badly;
      }
    }

    output_file << ' ';
    if (output_file.bad()) goto terminate_badly;

    if (has_qid && qid[row] != missing_qid)
    {
      output_file << "qid:" << qid[row] << ' ';
      if (output_file.bad()) goto terminate_badly;
    }

    ix1 = indptr[row];
    ix2 = indptr[row+1];
    n_this = ix2 - ix1;
    if (n_this == 1)
    {
      if (!ignore_zero_valued || above_min_decimals(values[ix1]))
      {
        output_file << (indices[ix1] + text_is_base1) << ':' << values[ix1];
        if (output_file.bad()) goto terminate_badly;
      }
    }

    else if (n_this > 1)
    {
      for (size_t ix = ix1; ix < ix2-1; ix++)
      {
        if (!ignore_zero_valued || above_min_decimals(values[ix]))
        {
          output_file << (indices[ix] + text_is_base1) << ':' << values[ix] << ' ';
          if (output_file.bad()) goto terminate_badly;
        }
      }
      if (!ignore_zero_valued || above_min_decimals(values[ix2-1]))
      {
        output_file << (indices[ix2-1] + text_is_base1) << ':' << values[ix2-1];
        if (output_file.bad()) goto terminate_badly;
      }
    }

    output_file << '\n';
    if (output_file.bad()) goto terminate_badly;
  }

  terminate_func:
  output_file.flags(oldflags);
  output_file.precision(oldprecision);
  return succeded;

  terminate_badly:
  succeded = false;
  goto terminate_func;
}

template <class label_t>
bool write_single_label_template
(
  FILE *output_file,
  int *restrict indptr,
  int *restrict indices,
  double *restrict values,
  label_t *restrict labels,
  int *restrict qid,
  const bool has_qid,
  const int nrows,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places
)
{
  if (output_file == NULL)
  {
    Rcpp::Rcerr << "Error: invalid output_file." << std::endl;
    return false;
  }

  if (decimal_places < 0)
  {
    Rcpp::Rcerr << "Error: 'decimal_places' cannot be negative." << std::endl;
    return false;
  }

  int succeded;
  if (add_header)
  {
    succeded = fprintf(output_file, "%d %d %d\n",
               nrows, ncols, nclasses);
    if (succeded < 0) return false;
  }

  if (nrows == 0)
    return true;

  if (sort_indices)
  {
    sort_sparse_indices(indptr, indices, values, nrows);
  }

  double thr_decimals = std::pow(10., (double)(-decimal_places)) / 2.;
  bool label_is_num = std::is_same<label_t, double>::value;

  char format_specifier_spaced[23];
  char format_specifier[23];
  char label_specifier[15];

  sprintf(format_specifier_spaced, "%%d:%%.%df ", decimal_places);
  sprintf(format_specifier, "%%d:%%.%df", decimal_places);
  if (label_is_num) {
    sprintf(label_specifier, "%%.%df ", decimal_places);
  } else {
    sprintf(label_specifier, "%s", "%d ");
  }

  size_t n_this, ix1, ix2;
  for (size_t row = 0; row < nrows; row++)
  {
    if (label_is_num)
    {
      if (!ISNAN(labels[row])) {
        succeded = fprintf(output_file, label_specifier, labels[row]);
      }

      else {
        succeded = fprintf(output_file, " ");
      }
    }

    else
    {
      if (labels[row] != NA_INTEGER) {
        succeded = fprintf(output_file, label_specifier, labels[row]);
      }

      else {
        succeded = fprintf(output_file, " ");
      }
    }

    if (succeded < 0) goto throw_err;

    if (has_qid && qid[row] != missing_qid)
    {
      succeded = fprintf(output_file, "qid:%d ", qid[row]);
      if (succeded < 0) goto throw_err;
    }

    ix1 = indptr[row];
    ix2 = indptr[row+1];
    n_this = ix2 - ix1;
    if (n_this == 1)
    {
      if (!ignore_zero_valued || above_min_decimals(values[ix1]))
      {
        succeded = fprintf(output_file, format_specifier,
                   indices[ix1] + text_is_base1, values[ix1]);
        if (succeded < 0) goto throw_err;
      }
    }

    else if (n_this > 1)
    {
      for (size_t ix = ix1; ix < ix2-1; ix++)
      {
        if (!ignore_zero_valued || above_min_decimals(values[ix]))
        {
          succeded = fprintf(output_file, format_specifier_spaced,
                     indices[ix] + text_is_base1, values[ix]);
          if (succeded < 0) goto throw_err;
        }
      }
      if (!ignore_zero_valued || above_min_decimals(values[ix2-1]))
      {
        succeded = fprintf(output_file, format_specifier,
                   indices[ix2-1] + text_is_base1, values[ix2-1]);
        if (succeded < 0) goto throw_err;
      }
    }

    succeded = fprintf(output_file, "\n");
    if (succeded < 0) goto throw_err;
  }

  return true;

  throw_err:
  {
    throw_errno(errno);
    return false;
  }
}

/* Do not remove this comment, it's a fix for an Rcpp bug */
// [[Rcpp::export(rng = false)]]
bool write_multi_label_R
(
  Rcpp::CharacterVector fname,
  Rcpp::IntegerVector indptr,
  Rcpp::IntegerVector indices,
  Rcpp::NumericVector values,
  Rcpp::IntegerVector indptr_lab,
  Rcpp::IntegerVector indices_lab,
  Rcpp::IntegerVector qid,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places,
  const bool append
)
{
  FILE *output_file = RC_fopen(fname[0], append? "a" : "w", TRUE);
  if (output_file == NULL)
  {
    throw_errno(errno);
    return false;
  }
  bool succeeded = write_multi_label_cpp(
    output_file,
    INTEGER(indptr),
    INTEGER(indices),
    REAL(values),
    INTEGER(indptr_lab),
    INTEGER(indices_lab),
    INTEGER(qid),
    qid.size() > 0,
    indptr.size() - 1,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    add_header,
    decimal_places
  );
  if (output_file != NULL) fclose(output_file);
  return succeeded;
}


// [[Rcpp::export(rng = false)]]
Rcpp::List write_multi_label_to_str_R
(
  Rcpp::IntegerVector indptr,
  Rcpp::IntegerVector indices,
  Rcpp::NumericVector values,
  Rcpp::IntegerVector indptr_lab,
  Rcpp::IntegerVector indices_lab,
  Rcpp::IntegerVector qid,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places
)
{
  Rcpp::List out = Rcpp::List::create(
    Rcpp::_["str"] = R_NilValue
  );
  std::stringstream ss;
  bool succeeded = write_multi_label_cpp(
    ss,
    INTEGER(indptr),
    INTEGER(indices),
    REAL(values),
    INTEGER(indptr_lab),
    INTEGER(indices_lab),
    INTEGER(qid),
    qid.size() > 0,
    indptr.size() - 1,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    add_header,
    decimal_places
  );
  if (!succeeded)
    return Rcpp::List();
  
  out["str"] = Rcpp::unwindProtect(convert_StringStreamToRcpp, (void*)&ss);
  return out;
}

// [[Rcpp::export(rng = false)]]
bool write_single_label_numeric_R
(
  Rcpp::CharacterVector fname,
  Rcpp::IntegerVector indptr,
  Rcpp::IntegerVector indices,
  Rcpp::NumericVector values,
  Rcpp::NumericVector labels,
  Rcpp::IntegerVector qid,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places,
  const bool append
)
{
  FILE *output_file = RC_fopen(fname[0], append? "a" : "w", TRUE);
  if (output_file == NULL)
  {
    throw_errno(errno);
    return false;
  }
  bool succeeded = write_single_label_template(
    output_file,
    INTEGER(indptr),
    INTEGER(indices),
    REAL(values),
    REAL(labels),
    INTEGER(qid),
    qid.size() > 0,
    indptr.size()-1,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    add_header,
    decimal_places
  );
  if (output_file != NULL) fclose(output_file);
  return succeeded;
}

// [[Rcpp::export(rng = false)]]
bool write_single_label_integer_R
(
  Rcpp::CharacterVector fname,
  Rcpp::IntegerVector indptr,
  Rcpp::IntegerVector indices,
  Rcpp::NumericVector values,
  Rcpp::IntegerVector labels,
  Rcpp::IntegerVector qid,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places,
  const bool append
)
{
  FILE *output_file = RC_fopen(fname[0], append? "a" : "w", TRUE);
  if (output_file == NULL)
  {
    throw_errno(errno);
    return false;
  }
  bool succeeded = write_single_label_template(
    output_file,
    INTEGER(indptr),
    INTEGER(indices),
    REAL(values),
    INTEGER(labels),
    INTEGER(qid),
    qid.size() > 0,
    indptr.size()-1,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    add_header,
    decimal_places
  );
  if (output_file != NULL) fclose(output_file);
  return succeeded;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List write_single_label_numeric_to_str_R
(
  Rcpp::IntegerVector indptr,
  Rcpp::IntegerVector indices,
  Rcpp::NumericVector values,
  Rcpp::NumericVector labels,
  Rcpp::IntegerVector qid,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places
)
{
  Rcpp::List out = Rcpp::List::create(
    Rcpp::_["str"] = R_NilValue
  );

  std::stringstream ss;
  bool succeeded = write_single_label_template(
    ss,
    INTEGER(indptr),
    INTEGER(indices),
    REAL(values),
    REAL(labels),
    INTEGER(qid),
    qid.size() > 0,
    indptr.size()-1,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    add_header,
    decimal_places
  );
  if (!succeeded)
    return Rcpp::List();

  out["str"] = Rcpp::unwindProtect(convert_StringStreamToRcpp, (void*)&ss);
  return out;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List write_single_label_integer_to_str_R
(
  Rcpp::IntegerVector indptr,
  Rcpp::IntegerVector indices,
  Rcpp::NumericVector values,
  Rcpp::IntegerVector labels,
  Rcpp::IntegerVector qid,
  const int ncols,
  const int nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool add_header,
  const int decimal_places
)
{
  Rcpp::List out = Rcpp::List::create(
    Rcpp::_["str"] = R_NilValue
  );

  std::stringstream ss;
  bool succeeded = write_single_label_template(
    ss,
    INTEGER(indptr),
    INTEGER(indices),
    REAL(values),
    INTEGER(labels),
    INTEGER(qid),
    qid.size() > 0,
    indptr.size()-1,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    add_header,
    decimal_places
  );
  if (!succeeded)
    return Rcpp::List();

  out["str"] = Rcpp::unwindProtect(convert_StringStreamToRcpp, (void*)&ss);
  return out;
}
