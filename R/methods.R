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
  if(is.logical(i)) i = which(i)
  if(is.character(i)) i = match(i, index_names)
  i[i < 0] = max_i + i[i < 0] + 1L
  if(anyNA(i) || any(i >  max_i, na.rm = TRUE))
    stop("some of row subset indices are not present in matrix")
  i
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
#' inherits(m[1, , drop=TRUE], "sparseVector")
#' inherits(m[1, 1, drop=TRUE], "numeric")
NULL

subset_csr = function(x, i, j, drop = TRUE) {

  if(missing(i) && missing(j)) return(x)

  row_names = rownames(x)
  col_names = colnames(x)

  if(missing(i)) i = seq_len(nrow(x))
  if(missing(j)) j = seq_len(ncol(x))
  # convert integer/numeric/logical/character indices to integer indices
  # also takes care of negatice indices
  i = get_indices_integer(i, nrow(x), row_names)
  j = get_indices_integer(j, ncol(x), col_names)

  n_row = length(i)
  n_col = length(j)
  col_indices = lapply(seq_len(n_row), function(x) integer())
  x_values = lapply(seq_len(n_row), function(x) numeric())

  for(k in seq_len(n_row) ) {
    j1 = x@p[[ i[[k]] ]]
    j2 = x@p[[ i[[k]] + 1L ]]
    if(j2 > j1) {
      j_seq = seq.int(j1, j2 - 1L) + 1L

      # indices should start with 1
      jj = x@j[j_seq] + 1L
      # FIXME may be it will make sense to replace with fastmatch::fmatch
      keep = match(jj, j, nomatch = 0L)
      # keep only those which are in requested columns
      which_keep = keep > 0L
      keep = keep[which_keep]

      # indices starting with 0
      col_indices[[k]] = keep - 1L

      x_values[[k]] = x@x[j_seq][which_keep]
    }
  }
  res = new("dgRMatrix")
  res@p = c(0L, cumsum(lengths(x_values)))
  res@j = do.call(c, col_indices)
  res@x = do.call(c, x_values)
  res@Dim = c(n_row, n_col)

  row_names = if(is.null(row_names)) NULL else row_names[i]
  col_names = if(is.null(col_names)) NULL else col_names[j]
  res@Dimnames = list(row_names, col_names)


  if(isTRUE(drop) && (n_row == 1L || n_col == 1L)) {
    if (n_row == 1L && n_col == 1L) {
      res = ifelse(NROW(res@x), res@x, 0)
    } else if (n_row == 1L) {
      res = Matrix::sparseVector(res@x, res@j+1L, length=n_col)
    } else {
      res = Matrix::sparseVector(res@x, convert_indptr_to_rows(res@p), length=n_row)
    }
  }
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
