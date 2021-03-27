#include "read_write.h"

#ifndef PRId64
#   if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#     define PRId64 "I64d"
#   else
#     define PRId64 "lld"
#   endif
#endif
#ifndef PRIu64
#   if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER))
#     define PRIu64 "I64u"
#   else
#     define PRId64 "llu"
#   endif
#endif
#ifndef SCNd64
#   define SCNd64 PRId64
#endif
#ifndef SCNu64
#   define SCNu64 PRIu64
#endif

/* https://stackoverflow.com/questions/16696297/ftell-at-a-position-past-2gb */
#if (defined(_WIN32) || defined(_WIN64) || defined(_MSC_VER)) && defined(PLATFORM_IS_64_OR_HIGHER)
#   ifdef _MSC_VER
#     include <limits> /* https://stackoverflow.com/questions/2561368/illegal-token-on-right-side-of */
#     define fseek_ _fseeki64
#     define ftell_ _ftelli64
#     define fpos_t_ __int64
#   elif defined(__GNUG__) || defined(__GNUC__)
#     define fseek_ fseeko
#     define ftell_ ftello
#     define fpos_t_ off_t
#   else
#     define fseek_ fseek
#     define ftell_ ftell
#     define fpos_t_ long /* <- might overflow with large files */
#   endif
#else
#   define fseek_ fseek
#   define ftell_ ftell
#   define fpos_t_ long
#endif

#define WHITESPACE_CHARS " \t\v"
#define missing_label NA_REAL

/* https://stackoverflow.com/questions/9103294/c-how-to-inspect-file-byte-order-mark-in-order-to-get-if-it-is-utf-8 */
/* sequences to skip:
0x00, 0x00, 0xfe, 0xff
0xff, 0xfe, 0x00, 0x00
0xfe, 0xff
0xff, 0xfe
0xef, 0xbb, 0xbf
 */
void skip_bom(FILE *input_file)
{
  fpos_t_ return_to = ftell_(input_file);
  int next_char = fgetc(input_file);
  char ch1, ch2, ch3, ch4;
  if (next_char == EOF) goto rewind;
  ch1 = next_char;
  if (ch1 == '\x00' || ch1 == '\xFF' ||
    ch1 == '\xFE' || ch1 == '\xEF')
  {
    ch1 = next_char;
    next_char = fgetc(input_file);
    if (next_char == EOF) goto rewind;
    ch2 = next_char;
    if (
      (ch1 == '\x00' && ch2 == '\x00') ||
      (ch1 == '\xFF' && ch2 == '\xFE') ||
      (ch1 == '\xFE' && ch2 == '\xFF') ||
      (ch1 == '\xFF' && ch2 == '\xFE') ||
      (ch1 == '\xEF' && ch2 == '\xBB')
    )
    {
      if (
        (ch1 == '\xFE' && ch2 == '\xFF') ||
        (ch1 == '\xFF' && ch2 == '\xFE')
      ) {
        return;
      }

      next_char = fgetc(input_file);
      if (next_char == EOF) goto rewind;
      ch3 = next_char;
      if (
        (ch1 == '\xEF' && ch2 == '\xBB' && ch3 == '\xBF')
      ) {
        return;
      }

      if (ch3 == '\xFE' || ch3 == '\x00')
      {
        next_char = fgetc(input_file);
        if (next_char == EOF) goto rewind;
        ch4 = next_char;
        if (
          (ch1 == '\x00' && ch2 == '\x00' && ch3 == '\xFE' && ch4 == '\xFF') ||
          (ch1 == '\xFF' && ch2 == '\xFE' && ch3 == '\x00' && ch4 == '\x00')
        ) {
          return;
        }
      }
    }
  }

  rewind:
  {
    fseek_(input_file, return_to, SEEK_SET);
    return;
  }
}

void subtract_one_from_vec(std::vector<int> &vec)
{
  for (auto el : vec)
    if (el <= 0)
      return;
  for (size_t ix = 0; ix < vec.size(); ix++)
    vec[ix] -= 1;
}

int find_largest_val(std::vector<int> &vec, int missing_label)
{
  int max_val = 0;
  for (int el : vec)
    max_val = std::max(max_val, (el != missing_label)? el : max_val);
  return max_val;
}


bool read_multi_label_cpp
(
  std::istream &input_file,
  std::vector<int> &indptr,
  std::vector<int> &indices,
  std::vector<double> &values,
  std::vector<int> &indptr_lab,
  std::vector<int> &indices_lab,
  std::vector<int> &qid,
  int &nrows,
  int &ncols,
  int &nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool assume_no_qid
)
{
  if (input_file.fail())
  {
    Rcpp::Rcerr << "Error: cannot open input file path." << std::endl;
    return false;
  }
  std::string ln;

  indptr.clear();
  indices.clear();
  values.clear();
  indptr_lab.clear();
  indices_lab.clear();

  indptr.push_back(0);
  indptr_lab.push_back(0);

  bool is_first_line = true;

  int curr_lab = 0;
  long bytes_advance = 0, adv1 = 0, adv2 = 0;
  char *ln_char;

  int curr_col = 0;
  double curr_val = 0;

  size_t lim_first;
  long long remainder;

  int header_nrows = 0, header_ncols = 0, header_nclasses = 0;

  while (std::getline(input_file, ln))
  {
    if (is_first_line && std::regex_search(ln, std::regex("^\\s*\\d+\\s+\\d+\\s+\\d+")))
    {
      uint64_t temp1, temp2, temp3;
      std::sscanf(ln.c_str(), "%" SCNu64 " %" SCNu64 " %" SCNu64,
            &temp1, &temp2, &temp3);
      bool size_is_within_type
        =
      temp1 < INT_MAX && temp2 < INT_MAX && temp3 < INT_MAX;
      header_nrows = temp1;
      header_ncols = temp2;
      header_nclasses = temp3;

      if (!size_is_within_type)
        return false;
      indptr.reserve(header_nrows);
      indptr_lab.reserve(header_nrows);
      if (!assume_no_qid) qid.reserve(header_nrows);
      is_first_line = false;
      continue;
    }
    is_first_line = false;

    if ((ln.size() <= 1 && (ln.size() == 0 || ln[0] != '#')) ||
      (ln.size() == 2 && ln[0] == '\r'))
    {
      indptr_lab.push_back(indices_lab.size());
      indptr.push_back(indices.size());
      if (!assume_no_qid)
        qid.push_back(missing_qid);
      continue;
    }

    ln_char = (char*)ln.c_str();
    if (ln_char[0] == '#')
      continue;
    lim_first = ln.find_first_of(WHITESPACE_CHARS, 0);
    if (lim_first == std::string::npos) {
      if (ln[0] == '\r')
        continue;
      else {
        Rcpp::Rcerr << "Invalid line encountered at row " << indptr.size() << std::endl;
        return false;
      }
    }
    remainder = lim_first;
    if (remainder == 0)
      goto get_features;
    
    adv2 = 0;
    while (sscanf(ln_char, "%d%ln,%ln", &curr_lab, &adv1, &adv2) == 1)
    {
      bytes_advance = adv1 + (bool)adv2;
      indices_lab.push_back(curr_lab);
      ln_char += bytes_advance;
      remainder -= bytes_advance;
      if (remainder <= 0)
        break;
    }

    get_features:
    indptr_lab.push_back(indices_lab.size());
    ln_char = (char*)ln.c_str() + lim_first + 1;

    if (!assume_no_qid)
    {
      auto lim_next = ln.find("qid:", lim_first + 1);
      if (lim_next == std::string::npos)
        qid.push_back(missing_qid);
      else {
        auto pos_comment = ln.find('#', lim_next + 1);
        auto n_matched = sscanf((char*)ln.c_str() + lim_next, "qid:%d%ln", &curr_col, &bytes_advance);
        if ((n_matched == 1) && (pos_comment == std::string::npos || pos_comment >= lim_next + bytes_advance)) {
          qid.push_back(curr_col);
          lim_first = ln.find_first_of(WHITESPACE_CHARS, lim_next + bytes_advance);
          if (lim_first == std::string::npos)
            goto next_line;
          ln_char = (char*)ln.c_str() + lim_first + 1;
        }
        else
          qid.push_back(missing_qid);
      }

    }

    remainder = lim_first;
    lim_first = ln.find('#', lim_first);
    if (lim_first == std::string::npos)
      remainder = ln.size() - remainder;
    else
      remainder = lim_first - remainder;
    if (remainder == 0)
      goto next_line;

    while (sscanf(ln_char, "%d:%lg%ln", &curr_col, &curr_val, &bytes_advance) == 2)
    {
      if (!ignore_zero_valued || curr_val)
      {
        indices.push_back(curr_col);
        values.push_back(curr_val);
      }
      ln_char += bytes_advance;
      remainder -= bytes_advance;
      if (remainder <= 0)
        break;
    }


    next_line:
    indptr.push_back(indices.size());
  }

  sort_sparse_indices(indptr, indices, values);
  std::vector<double> unused_vec;
  sort_sparse_indices(indptr_lab, indices_lab, unused_vec);

  if (text_is_base1) {
    subtract_one_from_vec(indices);
    subtract_one_from_vec(indices_lab);
  }

  nrows = indptr.size() - 1;
  int missing_ind = -1;
  ncols = std::max(header_ncols, find_largest_val(indices, missing_ind)+1);
  nclasses = std::max(header_nclasses, find_largest_val(indices_lab, missing_ind)+1);

  return true;
}


bool read_multi_label_cpp
(
  FILE *input_file,
  std::vector<int> &indptr,
  std::vector<int> &indices,
  std::vector<double> &values,
  std::vector<int> &indptr_lab,
  std::vector<int> &indices_lab,
  std::vector<int> &qid,
  int &nrows,
  int &ncols,
  int &nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool assume_no_qid
)
{
  if (input_file == NULL)
  {
    Rcpp::Rcerr << "Error: cannot open input file path." << std::endl;
    return false;
  }

  skip_bom(input_file);

  indptr.clear();
  indices.clear();
  values.clear();
  indptr_lab.clear();
  indices_lab.clear();

  indptr.push_back(0);
  indptr_lab.push_back(0);

  bool is_first_line = true;

  int curr_lab = 0;
  int curr_col = 0;
  double curr_val = 0;

  fpos_t_ return_to = ftell_(input_file);
  int n_matched;
  int next_char = (int)'a';

  int header_nrows = 0, header_ncols = 0, header_nclasses = 0;

  while (true)
  {
    if (is_first_line)
    {
      char buffer_first_line[1000];
      n_matched = fscanf(input_file, "%999[^\n]", buffer_first_line);
      if (n_matched == EOF)
        break;
      uint64_t temp1, temp2, temp3;
      n_matched = sscanf(buffer_first_line, "%" SCNu64 " %" SCNu64 " %" SCNu64,
                 &temp1, &temp2, &temp3);
      header_nrows = temp1;
      header_ncols = temp2;
      header_nclasses = temp3;

      if (n_matched != 3)
      {
        header_nrows = 0;
        header_ncols = 0;
        header_nclasses = 0;
        fseek_(input_file, return_to, SEEK_SET);
        is_first_line = false;
        continue;
      }

      bool size_is_within_type
        =
      temp1 < INT_MAX && temp2 < INT_MAX && temp3 < INT_MAX;
      if (!size_is_within_type)
        return false;
      indptr.reserve(header_nrows);
      indptr_lab.reserve(header_nrows);
      if (!assume_no_qid) qid.reserve(header_nrows);
      is_first_line = false;
      do { next_char = fgetc(input_file); }
      while ((char)next_char != '\n' && next_char != EOF);
      if (next_char == EOF)
        break;
      continue;
    }
    is_first_line = false;

    
    return_to = ftell_(input_file);
    
    /* check for empty line */
    next_char = fgetc(input_file);
    if (isspace((char)next_char) || (char)next_char == '#' || next_char == EOF)
    {
      if (next_char == EOF) {
        break;
      }
      else if ((char)next_char == '\n' || (char)next_char == '\r') {
        indptr_lab.push_back(indices_lab.size());
        indptr.push_back(indices.size());
        if (!assume_no_qid) qid.push_back(missing_qid);
        while ((char)next_char != '\n' && next_char != EOF)
        { next_char = fgetc(input_file); };
        if (next_char == EOF)
          break;
        continue;
      }
      else if ((char)next_char == '#') {
        do { next_char = fgetc(input_file); }
        while ((char)next_char != '\n' && next_char != EOF);
        if (next_char == EOF)
          break;
        continue;
      }
      else {
        next_char = fgetc(input_file);
        if ((next_char >= 48 && next_char <= 57) ||
          (next_char >= 43 && next_char <= 46) ||
          ((char)next_char == 'i' || (char)next_char == 'I') ||
          ((char)next_char == 'n' || (char)next_char == 'N'))
        {
          /* 0-9, -, ., +, i(nf), n(an) (expected case) */
          fseek_(input_file, return_to, SEEK_SET);
        }

        else if (next_char == EOF) {
          break;
        }

        else {
          goto get_features;
        }
      }
    }

    else
    {
      fseek_(input_file, return_to, SEEK_SET);
    }

    while ((n_matched = fscanf(input_file, "%d[^:],", &curr_lab)) == 1)
    {
      next_char = fgetc(input_file);
      if ((char)next_char == ':')
      {
        fseek_(input_file, return_to, SEEK_SET);
        goto get_features;
      }
      indices_lab.push_back(curr_lab);
      return_to = ftell_(input_file);
      if ((char)next_char != ',')
        break;
    }


    get_features:
    indptr_lab.push_back(indices_lab.size());
    if (n_matched == EOF)
    {
      if (!assume_no_qid) qid.push_back(missing_qid);
      break;
    }
    if (indptr_lab.back() == indptr_lab[indptr_lab.size()-2])
    {
      fseek_(input_file, return_to, SEEK_SET);
    }

    if ((char)next_char == '\n' || (char)next_char == '\r' || (char)next_char == '#')
    {
      if (!assume_no_qid) qid.push_back(missing_qid);
      goto next_line;
    }
    return_to = ftell_(input_file);
    next_char = fgetc(input_file);
    if ((char)next_char == '\n' || (char)next_char == '\r' || (char)next_char == '#' || next_char == EOF) {
      if (!assume_no_qid) qid.push_back(missing_qid);
      goto next_line;
    }
    else
      fseek_(input_file, return_to, SEEK_SET);

    if (!assume_no_qid)
    {
      n_matched = fscanf(input_file, "qid:%d", &curr_col);
      if (n_matched == EOF) {
        qid.push_back(missing_qid);
        break;
      }
      else if (n_matched == 0)
        qid.push_back(missing_qid);
      else
        qid.push_back(curr_col);
    }

    while ((n_matched = fscanf(input_file, "%d:%lg[^#]", &curr_col, &curr_val)) == 2)
    {
      if (!ignore_zero_valued || curr_val)
      {
        indices.push_back(curr_col);
        values.push_back(curr_val);
      }
      next_char = fgetc(input_file);
      if ((char)next_char == '\n' || (char)next_char == '\r' || !isspace((char)next_char))
        break;
    }


    next_line:
    indptr.push_back(indices.size());
    if (indptr.back() == indptr[indptr.size()-2])
    {
      fseek_(input_file, return_to, SEEK_SET);
      do { next_char = fgetc(input_file); }
      while ((char)next_char != '\n' && next_char != EOF);
    }

    if (n_matched == EOF || next_char == EOF)
      break;
    else if ((char)next_char == '\n')
      continue;
    else {
      do { next_char = fgetc(input_file); }
      while ((char)next_char != '\n' && next_char != EOF);
      if (next_char == EOF)
        break;
    }
  }

  sort_sparse_indices(indptr, indices, values);
  std::vector<double> temp;
  sort_sparse_indices(indptr_lab, indices_lab, temp);

  if (text_is_base1) {
    subtract_one_from_vec(indices);
    subtract_one_from_vec(indices_lab);
  }

  nrows = indptr.size() - 1;
  int missing_ind = -1;
  ncols = std::max(header_ncols, find_largest_val(indices, missing_ind)+1);
  nclasses = std::max(header_nclasses, find_largest_val(indices_lab, missing_ind)+1);

  return true;
}

bool read_single_label_cpp
(
  std::istream &input_file,
  std::vector<int> &indptr,
  std::vector<int> &indices,
  std::vector<double> &values,
  std::vector<double> &labels,
  std::vector<int> &qid,
  int &nrows,
  int &ncols,
  int &nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool assume_no_qid
)
{
  if (input_file.fail())
  {
    Rcpp::Rcerr << "Error: cannot open input file path." << std::endl;
    return false;
  }
  std::string ln;

  indptr.clear();
  indices.clear();
  values.clear();
  labels.clear();

  indptr.push_back(0);

  bool is_first_line = true;

  double curr_lab = 0;
  long bytes_advance = 0;
  char *ln_char;

  int curr_col = 0;
  double curr_val = 0;

  size_t lim_first;
  long long remainder;

  int header_nrows = 0, header_ncols = 0, header_nclasses = 0;

  while (std::getline(input_file, ln))
  {
    if (is_first_line && std::regex_search(ln, std::regex("^\\s*\\d+\\s+\\d+\\s+\\d+")))
    {
      uint64_t temp1, temp2, temp3;
      std::sscanf(ln.c_str(), "%" SCNu64 " %" SCNu64 " %" SCNu64,
            &temp1, &temp2, &temp3);
      bool size_is_within_type
        =
      temp1 < INT_MAX && temp2 < INT_MAX;
      header_nrows = temp1;
      header_ncols = temp2;
      header_nclasses = temp3;
      if (!size_is_within_type)
        return false;
      indptr.reserve(header_nrows);
      labels.reserve(header_nrows);
      if (!assume_no_qid) qid.reserve(header_nrows);
      is_first_line = false;
      continue;
    }
    is_first_line = false;

    if ((ln.size() <= 1 && (ln.size() == 0 || ln[0] != '#')) ||
      (ln.size() == 2 && ln[0] == '\r'))
    {
      labels.push_back(missing_label);
      indptr.push_back(indices.size());
      if (!assume_no_qid)
        qid.push_back(missing_qid);
      continue;
    }

    ln_char = (char*)ln.c_str();
    if (ln_char[0] == '#')
      continue;
    lim_first = ln.find_first_of(WHITESPACE_CHARS, 0);
    if (lim_first == std::string::npos) {
      if (ln[0] == '\r')
        continue;
      else {
        Rcpp::Rcerr << "Invalid line encountered at row " << indptr.size() << std::endl;
        return false;
      }
    }
    remainder = lim_first;
    if (remainder == 0) {
      labels.push_back(missing_label);
      goto get_features;
    }
    
    if (sscanf(ln_char, "%lg%ln", &curr_lab, &bytes_advance) == 1)
      labels.push_back(curr_lab);
    else
      labels.push_back(missing_label);

    get_features:
    ln_char = (char*)ln.c_str() + lim_first + 1;

    if (!assume_no_qid)
    {
      auto lim_next = ln.find("qid:", lim_first + 1);
      if (lim_next == std::string::npos)
        qid.push_back(missing_qid);
      else {
        auto pos_comment = ln.find('#', lim_next + 1);
        auto n_matched = sscanf((char*)ln.c_str() + lim_next, "qid:%d%ln", &curr_col, &bytes_advance);
        if ((n_matched == 1) && (pos_comment == std::string::npos || pos_comment >= lim_next + bytes_advance)) {
          qid.push_back(curr_col);
          lim_first = ln.find_first_of(WHITESPACE_CHARS, lim_next + bytes_advance);
          if (lim_first == std::string::npos)
            goto next_line;
          ln_char = (char*)ln.c_str() + lim_first + 1;
        }
        else
          qid.push_back(missing_qid);
      }

    }

    remainder = lim_first;
    lim_first = ln.find('#', lim_first);
    if (lim_first == std::string::npos)
      remainder = ln.size() - remainder;
    else
      remainder = lim_first - remainder;
    if (remainder == 0)
      goto next_line;

    while (sscanf(ln_char, "%d:%lg%ln", &curr_col, &curr_val, &bytes_advance) == 2)
    {
      if (!ignore_zero_valued || curr_val)
      {
        indices.push_back(curr_col);
        values.push_back(curr_val);
      }
      ln_char += bytes_advance;
      remainder -= bytes_advance;
      if (remainder <= 0)
        break;
    }


    next_line:
    indptr.push_back(indices.size());
  }

  sort_sparse_indices(indptr, indices, values);
  if (text_is_base1)
    subtract_one_from_vec(indices);

  nrows = indptr.size() - 1;
  int missing_ind = -1;
  ncols = std::max(header_ncols, find_largest_val(indices, missing_ind)+1);
  nclasses = 0;

  return true;
}

bool read_single_label_cpp
(
  FILE *input_file,
  std::vector<int> &indptr,
  std::vector<int> &indices,
  std::vector<double> &values,
  std::vector<double> &labels,
  std::vector<int> &qid,
  int &nrows,
  int &ncols,
  int &nclasses,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool assume_no_qid
)
{
  if (input_file == NULL)
  {
    Rcpp::Rcerr << "Error: cannot open input file path." << std::endl;
    return false;
  }

  skip_bom(input_file);

  indptr.clear();
  indices.clear();
  values.clear();
  labels.clear();

  indptr.push_back(0);

  bool is_first_line = true;

  int header_nrows = 0, header_ncols = 0, header_nclasses = 0;

  double curr_lab = 0;
  int curr_col = 0;
  double curr_val = 0;

  fpos_t_ return_to = ftell_(input_file);
  int n_matched;
  int next_char = (int)'a';

  while (true)
  {
    if (is_first_line)
    {
      char buffer_first_line[1000];
      n_matched = fscanf(input_file, "%999[^\n]", buffer_first_line);
      if (n_matched == EOF)
        break;
      uint64_t temp1, temp2, temp3;
      n_matched = sscanf(buffer_first_line, "%" SCNu64 " %" SCNu64 " %" SCNu64,
                 &temp1, &temp2, &temp3);
      header_nrows = temp1;
      header_ncols = temp2;
      header_nclasses = temp3;

      if (n_matched != 3)
      {
        header_nrows = 0;
        header_ncols = 0;
        header_nclasses = 0;
        fseek_(input_file, return_to, SEEK_SET);
        is_first_line = false;
        continue;
      }

      bool size_is_within_type
        =
      temp1 < INT_MAX && temp2 < INT_MAX;
      if (!size_is_within_type)
        return false;
      indptr.reserve(header_nrows);
      labels.reserve(header_nrows);
      if (!assume_no_qid) qid.reserve(header_nrows);
      is_first_line = false;
      do { next_char = fgetc(input_file); }
      while ((char)next_char != '\n' && next_char != EOF);
      if (next_char == EOF)
        break;
      continue;
    }
    is_first_line = false;

    
    return_to = ftell_(input_file);
    
    /* check for empty line */
    next_char = fgetc(input_file);
    if (isspace((char)next_char) || (char)next_char == '#' || next_char == EOF)
    {
      if (next_char == EOF) {
        break;
      }
      else if ((char)next_char == '\n' || (char)next_char == '\r') {
        labels.push_back(missing_label);
        indptr.push_back(indices.size());
        if (!assume_no_qid) qid.push_back(missing_qid);
        while ((char)next_char != '\n' && next_char != EOF)
        { next_char = fgetc(input_file); };
        if (next_char == EOF)
          break;
        continue;
      }
      else if ((char)next_char == '#') {
        do { next_char = fgetc(input_file); }
        while ((char)next_char != '\n' && next_char != EOF);
        if (next_char == EOF)
          break;
        continue;
      }
      else {
        next_char = fgetc(input_file);
        if ((next_char >= 48 && next_char <= 57) ||
          (next_char >= 43 && next_char <= 46) ||
          ((char)next_char == 'i' || (char)next_char == 'I') ||
          ((char)next_char == 'n' || (char)next_char == 'N'))
        {
          /* 0-9, -, ., +, i(nf), n(an) (expected case) */
          fseek_(input_file, return_to, SEEK_SET);
        }

        else if (next_char == EOF) {
          break;
        }

        else {
          labels.push_back(missing_label);
          goto get_features;
        }
      }
    }

    else
    {
      fseek_(input_file, return_to, SEEK_SET);
    }

    if ((n_matched = fscanf(input_file, "%lg[^:]", &curr_lab)) == 1)
    {
      next_char = fgetc(input_file);
      if ((char)next_char == ':')
      {
        fseek_(input_file, return_to, SEEK_SET);
        labels.push_back(missing_label);
        goto get_features;
      }
      /* in case of multi-label, should take only the first one */
      else if ((char)next_char == ',')
      {
        do { next_char = fgetc(input_file); }
        while ((char)next_char != '#' && next_char != EOF && !isspace((char)next_char));
      }
      labels.push_back(curr_lab);
      return_to = ftell_(input_file);
    }

    else if (n_matched == EOF) {
      if (!assume_no_qid) qid.push_back(missing_qid);
      break;
    }

    else {
      fseek_(input_file, return_to, SEEK_SET);
      labels.push_back(missing_label);
    }

    get_features:
    if ((char)next_char == '\n' || (char)next_char == '\r' || (char)next_char == '#') {
      if (!assume_no_qid) qid.push_back(missing_qid);
      goto next_line;
    }
    return_to = ftell_(input_file);
    next_char = fgetc(input_file);
    if ((char)next_char == '\n' || (char)next_char == '\r' || (char)next_char == '#' || next_char == EOF) {
      if (!assume_no_qid) qid.push_back(missing_qid);
      goto next_line;
    }
    else
      fseek_(input_file, return_to, SEEK_SET);

    if (!assume_no_qid)
    {
      n_matched = fscanf(input_file, "qid:%d", &curr_col);
      if (n_matched == EOF) {
        qid.push_back(missing_qid);
        break;
      }
      else if (n_matched == 0)
        qid.push_back(missing_qid);
      else
        qid.push_back(curr_col);
    }

    while ((n_matched = fscanf(input_file, "%d:%lg[^#]", &curr_col, &curr_val)) == 2)
    {
      if (!ignore_zero_valued || curr_val)
      {
        indices.push_back(curr_col);
        values.push_back(curr_val);
      }
      next_char = fgetc(input_file);
      if ((char)next_char == '\n' || (char)next_char == '\r' || !isspace((char)next_char))
        break;
    }


    next_line:
    indptr.push_back(indices.size());
    if (indptr.back() == indptr[indptr.size()-2])
    {
      fseek_(input_file, return_to, SEEK_SET);
      do { next_char = fgetc(input_file); }
      while ((char)next_char != '\n' && next_char != EOF);
    }

    if (n_matched == EOF || next_char == EOF)
      break;
    else if ((char)next_char == '\n')
      continue;
    else {
      do { next_char = fgetc(input_file); }
      while ((char)next_char != '\n' && next_char != EOF);
      if (next_char == EOF)
        break;
    }
  }

  sort_sparse_indices(indptr, indices, values);
  if (text_is_base1)
    subtract_one_from_vec(indices);

  nrows = indptr.size() - 1;
  int missing_ind = -1;
  ncols = std::max(header_ncols, find_largest_val(indices, missing_ind)+1);
  nclasses = 0;

  return true;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List read_multi_label_R
(
  Rcpp::CharacterVector fname,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool assume_no_qid
)
{
  Rcpp::List out = Rcpp::List::create(
    Rcpp::_["nrows"] = Rcpp::IntegerVector(1),
    Rcpp::_["ncols"] = Rcpp::IntegerVector(1),
    Rcpp::_["nclasses"] = Rcpp::IntegerVector(1),
    Rcpp::_["values"] = R_NilValue,
    Rcpp::_["indptr"] = R_NilValue,
    Rcpp::_["indices"] = R_NilValue,
    Rcpp::_["indptr_lab"] = R_NilValue,
    Rcpp::_["indices_lab"] = R_NilValue,
    Rcpp::_["qid"] = R_NilValue
  );

  std::vector<int> indptr, indices, indptr_lab, indices_lab;
  std::vector<double> values;
  std::vector<int> qid;
  int nrows, ncols, nclasses;

  FILE *input_file = RC_fopen(fname[0], "r", TRUE);
  if (input_file == NULL)
  {
    throw_errno(errno);
    return Rcpp::List();
  }
  bool succeeded = read_multi_label_cpp(
    input_file,
    indptr,
    indices,
    values,
    indptr_lab,
    indices_lab,
    qid,
    nrows,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    assume_no_qid
  );
  if (input_file != NULL) fclose(input_file);

  if (!succeeded)
    return Rcpp::List();

  if (nrows >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(1));
  else if (ncols >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(2));
  else if (nclasses >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(3));

  INTEGER(out["nrows"])[0] = (int)nrows;
  INTEGER(out["ncols"])[0] = (int)ncols;
  INTEGER(out["nclasses"])[0] = (int)nclasses;
  out["values"] = Rcpp::unwindProtect(convert_NumVecToRcpp, (void*)&values);

  values.clear();
  out["indptr"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indptr);
  indptr.clear();
  out["indices"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indices);
  indices.clear();
  out["indptr_lab"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indptr_lab);
  indptr_lab.clear();
  out["indices_lab"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indices_lab);
  indices_lab.clear();
  out["qid"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&qid);
  qid.clear();
  return out;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List read_multi_label_from_str_R
(
  Rcpp::CharacterVector file_as_str,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool assume_no_qid
)
{
  Rcpp::List out = Rcpp::List::create(
    Rcpp::_["nrows"] = Rcpp::IntegerVector(1),
    Rcpp::_["ncols"] = Rcpp::IntegerVector(1),
    Rcpp::_["nclasses"] = Rcpp::IntegerVector(1),
    Rcpp::_["values"] = R_NilValue,
    Rcpp::_["indptr"] = R_NilValue,
    Rcpp::_["indices"] = R_NilValue,
    Rcpp::_["indptr_lab"] = R_NilValue,
    Rcpp::_["indices_lab"] = R_NilValue,
    Rcpp::_["qid"] = R_NilValue
  );

  std::string file_as_str_cpp = Rcpp::as<std::string>(file_as_str);
  std::stringstream ss;
  ss.str(file_as_str_cpp);

  std::vector<int> indptr, indices, indptr_lab, indices_lab;
  std::vector<double> values;
  std::vector<int> qid;
  int nrows, ncols, nclasses;

  bool succeeded = read_multi_label_cpp(
    ss,
    indptr,
    indices,
    values,
    indptr_lab,
    indices_lab,
    qid,
    nrows,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    assume_no_qid
  );
  if (!succeeded)
    return Rcpp::List();

  if (nrows >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(1));
  else if (ncols >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(2));
  else if (nclasses >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(3));

  INTEGER(out["nrows"])[0] = (int)nrows;
  INTEGER(out["ncols"])[0] = (int)ncols;
  INTEGER(out["nclasses"])[0] = (int)nclasses;
  out["values"] = Rcpp::unwindProtect(convert_NumVecToRcpp, (void*)&values);

  values.clear();
  out["indptr"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indptr);
  indptr.clear();
  out["indices"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indices);
  indices.clear();
  out["indptr_lab"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indptr_lab);
  indptr_lab.clear();
  out["indices_lab"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indices_lab);
  indices_lab.clear();
  out["qid"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&qid);
  qid.clear();
  return out;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List read_single_label_R
(
  Rcpp::CharacterVector fname,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool assume_no_qid
)
{
  Rcpp::List out = Rcpp::List::create(
    Rcpp::_["nrows"] = Rcpp::IntegerVector(1),
    Rcpp::_["ncols"] = Rcpp::IntegerVector(1),
    Rcpp::_["nclasses"] = Rcpp::IntegerVector(1),
    Rcpp::_["values"] = R_NilValue,
    Rcpp::_["indptr"] = R_NilValue,
    Rcpp::_["indices"] = R_NilValue,
    Rcpp::_["labels"] = R_NilValue,
    Rcpp::_["qid"] = R_NilValue
  );

  std::vector<int> indptr, indices;
  std::vector<double> values, labels;
  std::vector<int> qid;
  int nrows, ncols, nclasses;

  FILE *input_file = RC_fopen(fname[0], "r", TRUE);
  if (input_file == NULL)
  {
    throw_errno(errno);
    return Rcpp::List();
  }
  bool succeeded = read_single_label_cpp(
    input_file,
    indptr,
    indices,
    values,
    labels,
    qid,
    nrows,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    assume_no_qid
  );
  if (input_file != NULL) fclose(input_file);
  
  if (!succeeded)
    return Rcpp::List();


  if (nrows >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(1));
  else if (ncols >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(2));
  else if (nclasses >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(3));

  INTEGER(out["nrows"])[0] = (int)nrows;
  INTEGER(out["ncols"])[0] = (int)ncols;
  INTEGER(out["nclasses"])[0] = (int)nclasses;
  out["values"] = Rcpp::unwindProtect(convert_NumVecToRcpp, (void*)&values);

  values.clear();
  out["indptr"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indptr);
  indptr.clear();
  out["indices"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indices);
  indices.clear();
  out["labels"] = Rcpp::unwindProtect(convert_NumVecToRcpp, (void*)&labels);
  labels.clear();
  out["qid"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&qid);
  qid.clear();
  return out;
}

// [[Rcpp::export(rng = false)]]
Rcpp::List read_single_label_from_str_R
(
  Rcpp::CharacterVector file_as_str,
  const bool ignore_zero_valued,
  const bool sort_indices,
  const bool text_is_base1,
  const bool assume_no_qid
)
{
  Rcpp::List out = Rcpp::List::create(
    Rcpp::_["nrows"] = Rcpp::IntegerVector(1),
    Rcpp::_["ncols"] = Rcpp::IntegerVector(1),
    Rcpp::_["nclasses"] = Rcpp::IntegerVector(1),
    Rcpp::_["values"] = R_NilValue,
    Rcpp::_["indptr"] = R_NilValue,
    Rcpp::_["indices"] = R_NilValue,
    Rcpp::_["labels"] = R_NilValue,
    Rcpp::_["qid"] = R_NilValue
  );

  std::string file_as_str_cpp = Rcpp::as<std::string>(file_as_str);
  std::stringstream ss;
  ss.str(file_as_str_cpp);

  std::vector<int> indptr, indices;
  std::vector<double> values, labels;
  std::vector<int> qid;
  int nrows, ncols, nclasses;

  bool succeeded = read_single_label_cpp(
    ss,
    indptr,
    indices,
    values,
    labels,
    qid,
    nrows,
    ncols,
    nclasses,
    ignore_zero_valued,
    sort_indices,
    text_is_base1,
    assume_no_qid
  );
  if (!succeeded)
    return Rcpp::List();

  if (nrows >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(1));
  else if (ncols >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(2));
  else if (nclasses >= INT_MAX - 1)
    return Rcpp::List::create(Rcpp::_["err"] = Rcpp::wrap(3));

  INTEGER(out["nrows"])[0] = (int)nrows;
  INTEGER(out["ncols"])[0] = (int)ncols;
  INTEGER(out["nclasses"])[0] = (int)nclasses;
  out["values"] = Rcpp::unwindProtect(convert_NumVecToRcpp, (void*)&values);

  values.clear();
  out["indptr"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indptr);
  indptr.clear();
  out["indices"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&indices);
  indices.clear();
  out["labels"] = Rcpp::unwindProtect(convert_NumVecToRcpp, (void*)&labels);
  labels.clear();
  out["qid"] = Rcpp::unwindProtect(convert_IntVecToRcpp, (void*)&qid);
  qid.clear();
  return out;
}
