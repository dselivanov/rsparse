#' Multithreaded Sparse-Dense Matrix Multiplication
#'
#' @description Multithreaded \code{\%*\%}, \code{crossprod}, \code{tcrossprod}
#' for sparse-dense matrix multiplication
#'
#' @details
#' Accelerates sparse-dense matrix multiplications using openmp. Applicable to the following pairs:
#' (\code{dgRMatrix}, \code{matrix}), (\code{matrix}, \code{dgRMatrix}),
#' (\code{dgCMatrix}, \code{matrix}), (\code{matrix}, \code{dgCMatrix}) combinations
#'
#' @param x,y
#' dense \code{matrix} and sparse
#'  \code{Matrix::RsparseMatrix} / \code{Matrix::CsparseMatrix} matrices.
#'
#' @return
#' A dense \code{matrix}
#'
#' @name matmult
#' @rdname matmult
#' @examples
#' library(Matrix)
#' data("movielens100k")
#' k = 10
#' nc = ncol(movielens100k)
#' nr = nrow(movielens100k)
#' x_nc = matrix(rep(1:k, nc), nrow = nc)
#' x_nr = t(matrix(rep(1:k, nr), nrow = nr))
#' csc = movielens100k
#' csr = as(movielens100k, "RsparseMatrix")
#' dense = as.matrix(movielens100k)
#' identical(csr %*% x_nc, dense %*% x_nc)
#' identical(x_nr %*% csc, x_nr %*% dense)
NULL

#' @rdname matmult
#' @export
setMethod("%*%", signature(x="dgRMatrix", y="matrix"), function(x, y) {
  # restore on exit
  n_thread = RhpcBLASctl::blas_get_num_procs()
  on.exit(RhpcBLASctl::blas_set_num_threads(n_thread))

  # set num threads to 1 in order to avoid thread contention between BLAS and openmp threads
  RhpcBLASctl::blas_set_num_threads(1L)



  check_dimensions_match(x, y)
  res = csr_dense_tcrossprod(x, t(y), getOption("rsparse_omp_threads", 1L))
  set_dimnames(res, rownames(x), colnames(y))
})

#' @rdname matmult
#' @export
setMethod("%*%", signature(x="dgRMatrix", y="float32"), function(x, y) {
  x %*% float::dbl(y)
})

#' @rdname matmult
#' @export
setMethod("%*%", signature(x="float32", y="dgRMatrix"), function(x, y) {
  float::dbl(x) %*% y
})

#' @rdname matmult
#' @export
setMethod("tcrossprod", signature(x="dgRMatrix", y="matrix"), function(x, y) {
  # restore on exit
  n_thread = RhpcBLASctl::blas_get_num_procs()
  on.exit(RhpcBLASctl::blas_set_num_threads(n_thread))

  # set num threads to 1 in order to avoid thread contention between BLAS and openmp threads
  RhpcBLASctl::blas_set_num_threads(1L)

  check_dimensions_match(x, y, y_transposed = TRUE)
  res = csr_dense_tcrossprod(x, y, getOption("rsparse_omp_threads", 1L))
  set_dimnames(res, rownames(x), rownames(y))
})

#' @rdname matmult
#' @export
setMethod("tcrossprod", signature(x="dgRMatrix", y="float32"), function(x, y) {
  tcrossprod(x, float::dbl(y))
})

#' @rdname matmult
#' @export
setMethod("%*%", signature(x="matrix", y="dgCMatrix"), function(x, y) {
  # restore on exit
  n_thread = RhpcBLASctl::blas_get_num_procs()
  on.exit(RhpcBLASctl::blas_set_num_threads(n_thread))

  # set num threads to 1 in order to avoid thread contention between BLAS and openmp threads
  RhpcBLASctl::blas_set_num_threads(1L)


  check_dimensions_match(x, y)
  res = dense_csc_prod(x, y, getOption("rsparse_omp_threads", 1L))
  set_dimnames(res, rownames(x), colnames(y))
})

#' @rdname matmult
#' @export
setMethod("%*%", signature(x="float32", y="dgCMatrix"), function(x, y) {
  float::dbl(x) %*% y
})

#' @rdname matmult
#' @export
setMethod("%*%", signature(x="dgCMatrix", y="float32"), function(x, y) {
  x %*% float::dbl(y)
})

#' @rdname matmult
#' @export
setMethod("crossprod", signature(x="matrix", y="dgCMatrix"), function(x, y) {
  # restore on exit
  n_thread = RhpcBLASctl::blas_get_num_procs()
  on.exit(RhpcBLASctl::blas_set_num_threads(n_thread))

  # set num threads to 1 in order to avoid thread contention between BLAS and openmp threads
  RhpcBLASctl::blas_set_num_threads(1L)


  x = t(x)
  check_dimensions_match(x, y)
  res = dense_csc_prod(x, y, getOption("rsparse_omp_threads", 1L))
  set_dimnames(res, rownames(x), colnames(y))
})

#' @rdname matmult
#' @export
setMethod("crossprod", signature(x="float32", y="dgCMatrix"), function(x, y) {
  crossprod(float::dbl(x), y)
})

get_indices_integer = function(i, max_i, index_names) {
  if(is.numeric(i)) i = as.integer(i)
  if(is.character(i)) i = match(i, index_names)
  if(is.logical(i)) {
    if (length(i) != max_i) {
      i = seq(1L, max_i)[i]
    } else {
      i = which(i)
    }
  }
  if (any(i < 0))
    i = seq(1L, max_i)[i]
  if(anyNA(i) || any(i >  max_i, na.rm = TRUE))
    stop("some of row subset indices are not present in matrix")
  as.integer(i)
}

#' CSR Matrices Slicing
#'
#' @description natively slice CSR matrices without converting them to triplet/CSC
#'
#' @param x input \code{RsparseMatrix}
#' @param i row indices to subset
#' @param j column indices to subset
#' @param drop whether to simplify 1d matrix to a vector
#'
#' @return
#' A \code{RsparseMatrix}
#'
#' @name slice
#' @examples
#' library(Matrix)
#' library(rsparse)
#' # dgCMatrix - CSC
#' m = rsparsematrix(20, 20, 0.1)
#' # make CSR
#' m = as(m, "RsparseMatrix")
#' inherits(m[1:2, ], "RsparseMatrix")
#' inherits(m[1:2, 3:4], "RsparseMatrix")
NULL

subset_csr = function(x, i, j, drop = TRUE) {

  if(missing(i) && missing(j)) return(x)

  row_names = rownames(x)
  col_names = colnames(x)

  all_i = FALSE
  all_j = FALSE
  i_is_seq = FALSE
  j_is_seq = FALSE
  if (missing(j)) {
    all_j = TRUE
    j = seq_len(ncol(x))
    n_col = ncol(x)
  } else {
    j = get_indices_integer(j, ncol(x), col_names)
    if (length(j) == ncol(x) && j[1L] == 1L && j[length(j)] == ncol(x)) {
      if (all(j == seq(1L, ncol(x))))
        all_j = TRUE
    } else {
      j_is_seq = check_is_seq(j)
    }
    n_col = length(j)
  }
  if (missing(i)) {
    i = seq_len(nrow(x))
    all_i = TRUE
    i_is_seq = TRUE
    n_row = nrow(x)
  } else {
    # convert integer/numeric/logical/character indices to integer indices
    # also takes care of negative indices
    i = get_indices_integer(i, nrow(x), row_names)
    i_is_seq = check_is_seq(i)
    n_row = length(i)

    if (all_j && i_is_seq && length(i) == nrow(x) && i[1L] == 1L && i[length(i)] == nrow(x)) {
      if (all(i == seq(1L, nrow(x))))
        all_i = TRUE
    }
  }

  if (!NROW(i) || !NROW(j)) {
    res = new("dgRMatrix")
    res@p = integer(NROW(i) + 1L)
    res@Dim = c(NROW(i), NROW(j))

    row_names = if(is.null(row_names) || !NROW(row_names)) NULL else row_names[i]
    col_names = if(is.null(col_names) || !NROW(col_names)) NULL else col_names[j]
    res@Dimnames = list(row_names, col_names)
    return(res)
  }

  if (all_i && all_j) {
    return(x)
  } else if (length(x@x) == 0L) {
    indptr = integer(n_row + 1)
    col_indices = integer()
    x_values = numeric()
  } else if (i_is_seq && all_j) {
    first = x@p[i[1L]] + 1L
    last = x@p[i[n_row] + 1L]
    indptr = x@p[seq(i[1L], i[n_row]+1L)] - x@p[i[1L]]
    col_indices = x@j[first:last]
    x_values = x@x[first:last]
  } else if (!i_is_seq && all_j) {
    temp = copy_csr_rows(x@p, x@j, x@x, i-1L)
    indptr = temp$indptr
    col_indices = temp$indices
    x_values = temp$values
  } else if (j_is_seq) {
    temp = copy_csr_rows_col_seq(x@p, x@j, x@x, i-1L, j-1L)
    indptr = temp$indptr
    col_indices = temp$indices
    x_values = temp$values
  } else {
    temp = copy_csr_arbitrary(x@p, x@j, x@x, i-1L, j-1L)
    indptr = temp$indptr
    col_indices = temp$indices
    x_values = temp$values
  }

  res = new("dgRMatrix")
  res@p = indptr
  res@j = col_indices
  res@x = x_values
  res@Dim = c(n_row, n_col)

  row_names = if(is.null(row_names) || !NROW(row_names)) NULL else row_names[i]
  col_names = if(is.null(col_names) || !NCOL(col_names)) NULL else col_names[j]
  res@Dimnames = list(row_names, col_names)

  if(isTRUE(drop) && (n_row == 1L || n_col == 1L))
    res = as.vector(res)
  res
}

#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "index", j = "index", drop="logical"), subset_csr)
#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "missing", j = "index", drop="logical"), subset_csr)
#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "index", j = "missing", drop="logical"), subset_csr)
#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "missing", j = "missing", drop="logical"), subset_csr)


# nocov start
set_dimnames = function(target, new_rownames, new_colnames) {
  data.table::setattr(target, 'dimnames', list(new_rownames, new_colnames))
  invisible(target)
}

check_dimensions_match = function(x, y, y_transposed = FALSE) {
  test_non_conformable = if(y_transposed) {
    ncol(x) != ncol(y)
  } else {
    ncol(x) != nrow(y)
  }
  if(test_non_conformable) stop("non-conformable arguments")
}

# nocov end

#' Conversions between matrix types
#'
#' @description Convenience functions for converting to different sparse matrix formats.
#'
#' @details The functions internally use as(x, "?sparseMatrix"), so they might work
#' with other object classes if they register a conversion method for `Matrix` base
#' types.
#'
#' When passed a vector, the functions `as.csr.matrix` and `as.coo.matrix` will
#' assume that it is a row vector, while `as.csc.matrix` will assume it's a column vector.
#'
#' @param x A matrix which is to be converted to a different format.
#' @param binary Whether the result should be a binary-only matrix (inheriting from
#' class `nsparseMatrix` - these don't have slot `x`).
#' Supported input types are:\itemize{
#' \item Sparse matrices from `Matrix` package, in any format.
#' \item Sparse vector from `Matrix` (class `dsparseVector`).
#' \item Dense matrix from base R.
#' \item Dense vector from base R (classes `numeric` and `integer`).
#' \item Dense matrix or vector from package `float` (class `float32`).
#' \item `data.frame` and `data.table`.
#' }
#'
#' @return A sparse matrix, with format:\itemize{
#' \item CSR (a.k.a. `RsparseMatrix`) when calling `as.csr.matrix`
#' (class `dgRMatrix` with `binary=FALSE`, class `ngRMatrix` with `binary=TRUE`).
#' \item CSC (a.k.a. `CsparseMatrix`) when calling `as.csc.matrix`
#' (class `dgCMatrix` with `binary=FALSE`, class `ngCMatrix` with `binary=TRUE`).
#' \item COO (a.k.a. `TsparseMatrix`) when calling `as.coo.matrix`
#' (class `dgTMatrix` with `binary=FALSE`, class `ngTMatrix` with `binary=TRUE`).
#' }
#'
#' @name casting
#' @examples
#' library(Matrix)
#' library(rsparse)
#'
#' m.coo = as(matrix(1:3), "TsparseMatrix")
#' as.csr.matrix(m.coo)
#' as.csr.matrix(1:3) # <- assumes it's a row vector
#' as.csc.matrix(1:3) # <- assumes it's a column vector
#'
#' library(float)
#' m.f32 = float::fl(matrix(1:10, nrow=5))
#' as.csr.matrix(m.f32)
#'
#' library(data.table)
#' as.coo.matrix(data.table(col1=1:3))
NULL

#' @rdname casting
#' @export
as.csr.matrix = function(x, binary=FALSE) {
  if ((inherits(x, "dgRMatrix") && !binary) || (inherits(x, "ngRMatrix") && binary))
    return(x)

  if (inherits(x, "float32"))
    x = float::dbl(x)

  if (inherits(x, c("numeric", "integer")))
    x = matrix(x, nrow=1L)

  if (inherits(x, c("data.frame", "tibble", "data.table")))
    x = as.matrix(x)

  if (inherits(x, "dsparseVector")) {
    X.csr = new("dgRMatrix")
    X.csr@Dim = c(1L, x@length)
    X.csr@p = c(0L, length(x@i))
    X.csr@j = x@i - 1L
    X.csr@x = x@x
    x = X.csr
  }

  if (!inherits(x, "RsparseMatrix"))
    x = as(x, "RsparseMatrix")

  if (!binary && !inherits(x, "dgRMatrix")) {
    X.csr = new("dgRMatrix")
    X.csr@Dim = x@Dim
    X.csr@Dimnames = x@Dimnames
    X.csr@p = x@p
    X.csr@j = x@j
    if (.hasSlot(x, "x"))
      X.csr@x = as.numeric(x@x)
    else
      X.csr@x = rep(1., length(x@j))
    x = X.csr
  }

  if (binary && !inherits(x, "ngRMatrix")) {
    X.csr = new("ngRMatrix")
    X.csr@Dim = x@Dim
    X.csr@Dimnames = x@Dimnames
    X.csr@p = x@p
    X.csr@j = x@j
    x = X.csr
  }
  return(x)
}

#' @rdname casting
#' @export
as.csc.matrix = function(x, binary=FALSE) {
  if ((inherits(x, "dgCMatrix") && !binary) || (inherits(x, "ngCMatrix") && binary))
    return(x)

  if (inherits(x, "float32"))
    x = float::dbl(x)

  if (inherits(x, c("numeric", "integer", "data.frame", "tibble", "data.table")))
    x = as.matrix(x)

  if (!inherits(x, "CsparseMatrix"))
    x = as(x, "CsparseMatrix")

  if (!binary && !inherits(x, "dgCMatrix")) {
    X.csc = new("dgCMatrix")
    X.csc@Dim = x@Dim
    X.csc@Dimnames = x@Dimnames
    X.csc@p = x@p
    X.csc@i = x@i
    if (.hasSlot(x, "x"))
      X.csc@x = as.numeric(x@x)
    else
      X.csc@x = rep(1., length(x@i))
    x = X.csc
  }

  if (binary && !inherits(x, "ngCMatrix")) {
    X.csc = new("ngCMatrix")
    X.csc@Dim = x@Dim
    X.csc@Dimnames = x@Dimnames
    X.csc@p = x@p
    X.csc@i = x@i
    x = X.csc
  }
  return(x)
}

#' @rdname casting
#' @export
as.coo.matrix = function(x, binary=FALSE) {
  if ((inherits(x, "dgTMatrix") && !binary) || (inherits(x, "ngTMatrix") && binary))
    return(x)

  if (inherits(x, "float32"))
    x = float::dbl(x)

  if (inherits(x, c("numeric", "integer")))
    x = matrix(x, nrow=1L)

  if (inherits(x, c("data.frame", "tibble", "data.table")))
    x = as.matrix(x)

  if (inherits(x, "dsparseVector"))
    x = as.csr.matrix(x)

  if (!inherits(x, "TsparseMatrix"))
    x = as(x, "TsparseMatrix")

  if (!binary && !inherits(x, "dgTMatrix")) {
    X.coo = new("dgTMatrix")
    X.coo@Dim = x@Dim
    X.coo@Dimnames = x@Dimnames
    X.coo@i = x@i
    X.coo@j = x@j
    if (.hasSlot(x, "x"))
      X.coo@x = as.numeric(x@x)
    else
      X.coo@x = rep(1., length(x@j))
    x = X.coo
  }

  if (binary && !inherits(x, "ngTMatrix")) {
    X.coo = new("ngTMatrix")
    X.coo@Dim = x@Dim
    X.coo@Dimnames = x@Dimnames
    X.coo@i = x@i
    X.coo@j = x@j
    x = X.coo
  }
  return(x)
}
