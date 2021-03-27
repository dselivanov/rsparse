check.bool = function(var, name) {
  if (NROW(var) != 1)
    stop(sprintf("Error: %s must be a single boolean/logical.", name))
  if (typeof(var) != "logical")
    var = as.logical(var)
  if (is.na(var))
    stop(sprintf("%s cannot be NA.", name))
  return(var)
}

check.int = function(var, name) {
  if (NROW(var) != 1)
    stop(sprintf("Error: %s must be a single non-negative integer.", name))
  if (typeof(var) != "integer")
    var = as.integer(var)
  if (is.na(var))
    stop(sprintf("%s cannot be NA.", name))
  if (is.infinite(var))
    stop(sprintf("%s cannot be infinite.", name))
  if (var < 0L)
    stop(sprintf("%s must be non-negative."))
  return(var)
}

cast.df = function(df) {
  if (inherits(df, c("tibble", "data.table")))
    df = as.data.frame(df)
  if (inherits(df, "data.frame"))
    df = as.matrix(df)
  return(df)
}

cast.dense.vec = function(X) {
  if (inherits(X, c("numeric", "integer", "float32"))) {
    if (typeof(X) != "double")
      X = as.numeric(X)
    X = matrix(X, nrow=1L)
  }
  return(X)
}

cast.sparse.vec = function(X) {
  if (inherits(X, "dsparseVector")) {
    X.csr = new("dgRMatrix")
    X.csr@Dim = c(1L, X@length)
    X.csr@p = c(0L, length(X@x))
    X.csr@j = X@i - 1L
    X.csr@x = X@x
    X = X.csr
  }
  return(X)
}

check.for.overflow = function(lst) {
  if (length(lst) == 1L) {
    if ((lst$err == 1) && ("err" %in% names(lst)))
      stop("Number of rows exceeds INT_MAX.")
    else if (lst$err == 2)
      stop("Number of columns exceeds INT_MAX.")
    else if (lst$err == 3)
      stop("Number of classes exceeds INT_MAX.")
  }
}

#' @title Read Sparse Matrix from Text File
#' @description Read a labelled sparse CSR matrix in text format as used by libraries
#' such as SVMLight, LibSVM, ThunderSVM, LibFM, xLearn, XGBoost, LightGBM, and more.
#' 
#' The format is as follows:
#' 
#' \code{<label(s)> <column>:<value> <column>:<value> ...}
#' 
#' with one line per observation/row.
#' 
#' Example line (row):
#' 
#' \code{1 1:1.234 3:20}
#' 
#' This line denotes a row with label (target variable) equal to 1, a value
#' for the first column of 1.234, a value of zero for the second column (which is
#' missing), and a value of 20 for the third column.
#' 
#' The labels might be decimal (for regression), and each row might contain more
#' than one label (must be integers in this case), separated by commas \bold{without}
#' spaces inbetween - e.g.:
#' 
#' \code{1,5,10 1:1.234 3:20}
#' 
#' This line indicates a row with labels 1, 5, and 10 (for multi-class classification).
#' If the line has no labels, it should still include a space before the features.
#' 
#' 
#' The rows might additionally contain a `qid` parameter as used in ranking algorithms,
#' which should always lay inbetween the labels and the features and must be an integer - e.g.:
#' 
#' \code{1 qid:2 1:1.234 3:20}
#' 
#' 
#' The file might optionally contain a header as the first line with metadata
#' (number of rows, number of columns, number of classes). Presence of a header will be
#' automatically detected, and is recommended to include it for speed purposes. Datasets
#' from the extreme classification repository (see references) usually include such a header.
#' 
#' Lines might include comments, which start after a `#` character. Lines consisting
#' of only a `#` will be ignored. When reading from a file, such file might have a
#' BOM (information about encoding uses in Windows sytems), which will be automatically
#' skipped.
#' 
#' @details Note that this function:\itemize{
#' \item Will not make any checks for negative column indices.
#' \item Has a precision of C type `int` for column indices and integer labels
#' (the maximum value that this type can hold can be checked in `.Machine$integer.max`).
#' \item Will fill missing labels with NAs when passing `multilabel=FALSE`.
#' \item Will fill with zeros (empty values) the lines that are empty (that is,
#' they generate a row in the data), but will ignore (that is, will not generate
#' a row in the data) the lines that start with `#`.
#' }
#' 
#' If the file contains a header, and this header denotes a larger number of columns
#' or of labels than the largest index in the data, the resulting object will have
#' this dimension set according to the header. The third entry in the header (number
#' of classes/labels) will be ignored when passing `multilabel=FALSE`.
#' @param file Either a file path from which the data will be read, or a string
#' (`character` variable) containing the text from which the data will be read.
#' In the latter case, must pass `from_string=TRUE`.
#' @param multilabel Whether the input file can have multiple labels per observation.
#' If passing `multilabel=FALSE` and it turns out to have multiple labels, will only
#' take the first one for each row. If the labels are non-integers or have decimal point,
#' the results will be invalid.
#' @param has_qid Whether the input file has `qid` field (used for ranking). If passing
#' `FALSE` and the file does turns out to have `qid`, the features will not be read for any
#' observations.
#' @param integer_labels Whether to output the observation labels as integers.
#' @param index1 Whether the input file uses numeration starting at 1 for the column
#' numbers (and for the label numbers when passing `multilabel=TRUE`). This is usually
#' the case for files downloaded from the repositories in the references. The function
#' will check for whether any of the column indices is zero, and will ignore this
#' option if so (i.e. will assume it is `FALSE`).
#' @param sort_indices Whether to sort the indices of the columns after reading the data.
#' These should already be sorted in the files from the repositories in the references.
#' @param ignore_zeros Whether to avoid adding features which have a value of zero.
#' If the zeros are caused due to numerical rounding in the software that wrote the
#' input file, they can be post-processed by passing `ignore_zeros=FALSE` and then
#' something like `X@x[X@x == 0] = 1e-8`.
#' @param min_cols Minimum number of columns that the output `X` object should have,
#' in case some columns are all missing in the input data.
#' @param min_classes Minimum number of columns that the output `y` object should have,
#' in case some columns are all missing in the input data. Only used when passing
#' `multilabel=TRUE`.
#' @param from_string Whether to read the data from a string variable instead of a file.
#' If passing `from_string=TRUE`, then `file` is assumed to be a variable with the
#' data contents on it.
#' @return A list with the following entries:\itemize{
#' \item `X`: the features, as a CSR Matrix from package `Matrix` (class `dgRMatrix`).
#' \item `y`: the labels. If passing `multilabel=FALSE` (the default), will be a vector
#' (class `numeric` when passing `integer_labels=FALSE`, class `integer` when passing
#' `integer_labels=TRUE`), otherwise will be a binary CSR Matrix (class `ngRMatrix`).
#' \item `qid`: the query IDs used for ranking, as an integer vector.
#' This entry will \bold{only} be present when passing `has_qid=TRUE`.
#' }
#' These can be easily transformed to other sparse matrix types through e.g.
#' `X = as(X, "CsparseMatrix")`.
#' @seealso \link{write.sparse}
#' @export
#' @examples 
#' library(Matrix)
#' library(rsparse)
#' 
#' ### Example input file
#' "1 2:1.21 5:2.05
#' -1 1:0.45 3:0.001 4:-10" -> coded.matrix
#' 
#' r = read.sparse(coded.matrix, from_string=TRUE)
#' print(r)
#' 
#' ### Convert it back to text
#' recoded.matrix = write.sparse(file=NULL, X=r$X, y=r$y, to_string=TRUE)
#' cat(recoded.matrix)
#' 
#' ### Example with real file I/O
#' ## generate a random sparse matrix and labels
#' set.seed(1)
#' X = rsparsematrix(nrow=5, ncol=10, nnz=8)
#' y = rnorm(5)
#' 
#' ## save into a text file
#' temp_file = file.path(tempdir(), "matrix.txt")
#' write.sparse(temp_file, X, y, integer_labels=FALSE)
#' 
#' ## inspect the text file
#' cat(paste(readLines(temp_file), collapse="\n"))
#' 
#' ## read it back
#' r = read.sparse(temp_file)
#' print(r)
#' 
#' ### (Note that columns with all-zeros are discarded,
#' ###  this behavior can be avoided with 'add_header=TRUE')
#' @references Datasets in this format can be found here:\itemize{
#' \item LibSVM Data: \url{https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets}
#' \item Extreme Classification Repository: \url{http://manikvarma.org/downloads/XC/XMLRepository.html}
#' }
#' 
#' The format is also described at the SVMLight webpage: \url{http://svmlight.joachims.org}.
read.sparse = function(file, multilabel=FALSE, has_qid=FALSE, integer_labels=FALSE,
            index1=TRUE, sort_indices=TRUE, ignore_zeros=TRUE,
            min_cols=0L, min_classes=0L,
            from_string=FALSE) {
  multilabel    =  check.bool(multilabel, "multilabel")
  has_qid     =  check.bool(has_qid, "has_qid")
  integer_labels  =  check.bool(integer_labels, "integer_labels")
  index1      =  check.bool(index1, "index1")
  sort_indices  =  check.bool(sort_indices, "sort_indices")
  ignore_zeros  =  check.bool(ignore_zeros, "ignore_zeros")
  from_string   =  check.bool(from_string, "from_string")
  
  min_cols    =  check.int(min_cols, "min_cols")
  min_classes   =  check.int(min_classes, "min_classes")
  
  if (inherits(file, "connection")) {
    file = paste(readLines(file), collapse="\n")
    from_string = TRUE
  }
  
  if (!from_string) {
    if (!file.exists(file))
      stop("Error: file does not exist.")
    
    if (multilabel) {
      read_func = read_multi_label_R
    } else {
      read_func = read_single_label_R
    }
    
  } else {
    if (typeof(file) != "character")
      stop("Must pass a character/string variable as input.")
    
    if (multilabel) {
      read_func = read_multi_label_from_str_R
    } else {
      read_func = read_single_label_from_str_R
    }
  }
  
  r = read_func(
    file,
    ignore_zeros,
    sort_indices,
    index1,
    !has_qid
  )
  
  if (!length(r))
    stop("Error: could not read file successfully.")
  check.for.overflow(r)
  
  r$ncols   =  as.integer(max(c(r$ncols, min_cols)))
  r$nclasses  =  as.integer(max(c(r$nclasses, min_classes)))
  
  features    =  new("dgRMatrix")
  features@Dim  =  c(r$nrows, r$ncols)
  features@p  =  r$indptr
  features@j  =  r$indices
  features@x  =  r$values
  
  if (multilabel) {
    labels    =  new("ngRMatrix")
    labels@Dim  =  c(r$nrows, r$nclasses)
    labels@p  =  r$indptr_lab
    labels@j  =  r$indices_lab
  } else {
    labels = r$labels
    if (integer_labels)
      labels = as.integer(labels)
  }

  if (!has_qid && length(r$indptr) > 1L &&
    r$indptr[1L] == r$indptr[length(r$indptr)]) {
    warning("Data has empty features. Perhaps the file has 'qid' field?")
  }
  
  if (!has_qid)
    return(list(X = features, y = labels))
  else
    return(list(X = features, y = labels, qid = r$qid))
}

#' @title Write Sparse Matrix in Text Format
#' @description Write a labelled sparse matrix into text format as used by software
#' such as SVMLight, LibSVM, ThunderSVM, LibFM, xLearn, XGBoost, LightGBM, and others - i.e.:
#' 
#' \code{<labels(s)> <column:value> <column:value> ...}
#' 
#' For more information about the format and usage examples, see \link{read.sparse}.
#' 
#' Can write labels for regression, classification (binary, multi-class, and multi-label),
#' and ranking (with `qid`), but note that most software that makes use of this data
#' format supports only regression and binary classification.
#' @details Be aware that writing sparse matrices to text is not a lossless operation
#' - that is, some information might be lost due to numeric precision, and metadata such
#' as row and column names will not be saved. It is recommended to use `saveRDS` or similar
#' for saving data between R sessions.
#' 
#' The option `ignore_zeros` is implemented heuristically, by comparing
#' `abs(x) >= 10^(-decimal_places)/2`, which might not match exactly with
#' the rounding that is done implicitly in string conversions in the libc/libc++
#' functions - thus there might still be some corner cases of all-zeros written into
#' features if the (absolute) values are very close to the rounding threshold.
#' 
#' While R uses C `double` type for numeric values, most of the software that is able to
#' take input data in this format uses `float` type, which has less precision.
#' 
#' The function uses different code paths when writing to a file or to a string,
#' and there might be slight differences between the generated texts from them.
#' If any such difference is encountered, please submit a bug report in the
#' package's GitHub page.
#' @param file Output file path into which to write the data.
#' Will be ignored when passing `to_string=TRUE`.
#' @param X Sparse data to write. Can be a sparse matrix from package `Matrix`
#' (classes: `dgRMatrix`, `dgTMatrix`, `dgCMatrix`, `ngRMatrix`, `ngTMatrix`, `ngCMatrix`)
#' or from package `SparseM` (classes: `matrix.csr`, `matrix.coo`, `matrix.csc`),
#' or a dense matrix of all numeric values, passed either as a `matrix` or as a `data.frame`.
#' 
#' If `X` is a vector (classes `numeric`, `integer`, `dsparseVector`), will be assumed to
#' be a row vector and will thus write one row only.
#' 
#' Note that the data will be casted to `dgRMatrix` in any case.
#' @param y Labels for the data. Can be passed as a vector (`integer` or `numeric`)
#' if each observation has one label, or as a sparse or dense matrix (same format as `X`)
#' if each observation can have more than 1 label. In the latter case, only the non-missing
#' column indices will be written, while the values are ignored.
#' @param qid Secondary label information used for ranking algorithms. Must be an integer vector
#' if passed. Note that not all software supports this.
#' @param integer_labels Whether to write the labels as integers. If passing `FALSE`, they will
#' have a decimal point regardless of whether they are integers or not. If the file is meant
#' to be used for a classification algorithm, one should pass `TRUE` here (the default).
#' For multilabel classification, the labels will always be written as integers.
#' @param index1 Whether the column and label indices (if multi-label) should have numeration
#' starting at 1. Most software assumes this is `TRUE`.
#' @param sort_indices Whether to sort the indices of `X` (and of `y` if multi-label) before
#' writing the data. Note that this will cause in-place modifications if either `X` or `y`
#' are passed as CSR matrices from `Matrix` package.
#' @param ignore_zeros Whether to ignore (not write) features with a value of zero
#' after rounding to the specified decimal places.
#' @param add_header Whether to add a header with metadata as the first line (number of rows,
#' number of columns, number of classes). If passing `integer_label=FALSE` and `y` is a
#' vector, will write zero as the number of labels. This is not supported by most software.
#' @param decimal_places Number of decimal places to use for numeric values. All values
#' will have exactly this number of places after the decimal point. Be aware that values
#' are rounded and might turn to zeros (will be skipped by default) if they are too small
#' (one can do something like
#' `X@x = ifelse(X@x >= 0, pmin(X@x, 1e-8), pmax(X@x, -1e-8))` to avoid this).
#' @param append Whether to append text at the end of the file instead of overwriting or
#' creating a new file. Ignored when passing `to_string=TRUE`.
#' @param to_string Whether to write the result into a string (which will be returned
#' from the function) instead of into a file.
#' @return If passing `to_string=FALSE` (the default), will not return anything
#' (`invisible(NULL)`). If passing `to_string=TRUE`, will return a `character`
#' variable with the data contents written into it.
#' @seealso \link{read.sparse}
#' @export
write.sparse = function(file, X, y, qid=NULL, integer_labels=TRUE,
             index1=TRUE, sort_indices=TRUE, ignore_zeros=TRUE,
             add_header=FALSE, decimal_places=8L,
             append=FALSE, to_string=FALSE) {
  if (is.null(X) || is.null(y))
    stop("Must pass 'X' and 'y'.")
  if (NROW(y) != NROW(X))
    stop("'X' and 'y' must have the same number of rows.")
  if (!to_string) {
    if ((typeof(file) != "character") || NROW(file) == 0L)
      stop("'file' must be a character variable containing a file path.")
  }
  if (NROW(decimal_places) != 1L)
    stop("'decimal_places' must be a single integer.")
  decimal_places = as.integer(decimal_places)
  if (is.na(decimal_places) || is.infinite(decimal_places) || (decimal_places < 0L))
    stop("Invalid 'decimal_places'.")
  if (decimal_places > 20L) {
    warning("'decimal_places' is greater than 20.")
  }
  
  
  integer_labels  =  check.bool(integer_labels, "integer_labels")
  index1      =  check.bool(index1, "index1")
  sort_indices  =  check.bool(sort_indices, "sort_indices")
  ignore_zeros  =  check.bool(ignore_zeros, "ignore_zeros")
  add_header    =  check.bool(add_header, "add_header")
  
  X = as.csr.matrix(X)
  y = cast.df(y)

  if (nrow(X) == 1L) {
    y = cast.sparse.vec(y)
  } else if (inherits(y, "dsparseVector")) {
    y = as.numeric(y)
  }
  
  allowed_y_multi = c("matrix", "matrix.coo", "matrix.csr", "matrix.csc",
             "dgRMatrix", "dgCMatrix", "dgTMatrix",
             "ngRMatrix", "ngCMatrix", "ngTMatrix")
  allowed_y_single = c("integer", "numeric", "factor", "float32")
  allowed_y = c(allowed_y_multi, allowed_y_single)
  if (!inherits(y, allowed_y))
    stop(sprintf("Invalid 'y' - allowed types: %s", paste0(allowed_y, collapse=",")))
  if (inherits(y, allowed_y_multi)) {
    y_is_multi = TRUE
    y = as.csr.matrix(y, binary=TRUE)
  } else {
    if (inherits(y, "float32") && ncol(y) != 1L)
      stop("'float32' objects not supported for multi-label 'y'.")
    y_is_multi = FALSE
    if (integer_labels) {
      if (typeof(y) != "integer")
        y = as.integer(y)
    } else {
      if (typeof(y) != "double")
        y = as.numeric(y)
    }
  }
  
  if (!NROW(qid)) {
    qid = integer()
  } else {
    if (NROW(qid) != NROW(X))
      stop("'X' and 'qid' must have the same number of rows.")
    if (typeof(qid) != "integer")
      qid = as.integer(qid)
  }
  
  if (integer_labels) {
    if (add_header) {
      if (any(is.infinite(y)) || all(is.na(y))) {
        n_classes = 0L
      } else if (any(y < 0)) {
        n_classes = length(unique(y))
      } else {
        n_classes = max(y, na.rm=TRUE)
        if (0L %in% y)
          n_classes = n_classes + 1L
        n_classes = max(n_classes, 1L)
        if (is.na(n_classes) || is.infinite(n_classes))
          nclasses = 0L
        if (typeof(n_classes) != "integer")
          n_classes = as.integer(n_classes)
      }
    } else
      n_classes = 0L
  } else {
    n_classes = 0L
  }
  
  if (!to_string) {
    if (add_header && append && file.exists(file))
      warning("Warning: adding header to existing file with 'append=TRUE'.")
    
    if (y_is_multi) {
      success = write_multi_label_R(
        file,
        X@p,
        X@j,
        X@x,
        y@p,
        y@j,
        qid,
        ncol(X),
        ncol(y),
        ignore_zeros,
        sort_indices,
        index1,
        add_header,
        decimal_places,
        append
      )
    } else {
      if (integer_labels) {
        success = write_single_label_integer_R(
          file,
          X@p,
          X@j,
          X@x,
          y,
          qid,
          ncol(X),
          n_classes,
          ignore_zeros,
          sort_indices,
          index1,
          add_header,
          decimal_places,
          append
        )
      } else {
        n_classes = 0L
        success = write_single_label_numeric_R(
          file,
          X@p,
          X@j,
          X@x,
          y,
          qid,
          ncol(X),
          n_classes,
          ignore_zeros,
          sort_indices,
          index1,
          add_header,
          decimal_places,
          append
        )
      }
    }
    if (!success)
      stop("Error: file write failed.")
    return(invisible(NULL))
  } else {
    if (y_is_multi) {
      res = write_multi_label_to_str_R(
        X@p,
        X@j,
        X@x,
        y@p,
        y@j,
        qid,
        ncol(X),
        ncol(y),
        ignore_zeros,
        sort_indices,
        index1,
        add_header,
        decimal_places
      )
    } else {
      if (integer_labels) {
        res = write_single_label_integer_to_str_R(
          X@p,
          X@j,
          X@x,
          y,
          qid,
          ncol(X),
          n_classes,
          ignore_zeros,
          sort_indices,
          index1,
          add_header,
          decimal_places
        )
      } else {
        n_classes = 0L
        res = write_single_label_numeric_to_str_R(
          X@p,
          X@j,
          X@x,
          y,
          qid,
          ncol(X),
          n_classes,
          ignore_zeros,
          sort_indices,
          index1,
          add_header,
          decimal_places
        )
      }
    }
    
    if (!length(res))
      stop("Error: string write failed.")
    return(res$str)
  }
}
