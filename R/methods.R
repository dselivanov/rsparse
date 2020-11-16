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
  i[i < 0] = max_i + i[i < 0] + 1L
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
