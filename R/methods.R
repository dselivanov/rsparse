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

#' @name operators
#' @title Mathematical operators on CSR matrices
#' @description Implements some mathematical operators for CSR matrices
#' (a.k.a. RsparseMatrix) for cases in which the right-hand side is a constant
#' and the operation reduces to applying the operator elementwise on the non-zero
#' entries only, doing so without converting the matrix to CSC in the process.
#' @param e1 A CSR matrix.
#' @param e2 Right-hand side of the operation. Must be a scalar (otherwise the
#' matrix will get converted to CSC and will call the corresponding operator
#' from the `Matrix` package.)
#' @return A CSR matrix of class `dgRMatrix` (`lgRMatrix` for some of the logical
#' operators).
#' @examples
#' library(Matrix)
#' library(rsparse)
#' set.seed(1)
#' X = as.csr.matrix(rsparsematrix(4, 3, .5))
#' X + X
#' X * 2
#' X ^ 2
#' ### here the result will be CSC
#' X ^ c(1,2)
NULL

multiply_csr_by_csr = function(e1, e2) {
  if (nrow(e1) != nrow(e2) || ncol(e1) != ncol(e2))
    stop("Matrices must have the same dimensions in order to multiply them.")

  if (inherits(e1, "nsparseMatrix") && inherits(e2, "nsparseMatrix")) {
    if (is_same_ngRMatrix(e1@p, e2@p, e1@j, e2@j))
      return(e1)
  }
  e1 = as.csr.matrix(e1)
  e2 = as.csr.matrix(e2)
  res = multiply_csr_elemwise(e1@p, e2@p, e1@j, e2@j, e1@x, e2@x)
  out = new("dgRMatrix")
  out@Dim = e1@Dim
  out@Dimnames = e1@Dimnames
  out@p = res$indptr
  out@j = res$indices
  out@x = res$values
  return(out)
}

#' @rdname operators
#' @export
setMethod("*", signature(e1="RsparseMatrix", e2="sparseMatrix"), multiply_csr_by_csr)

#' @rdname operators
#' @export
setMethod("*", signature(e1="ngRMatrix", e2="sparseMatrix"), multiply_csr_by_csr)

#' @rdname operators
#' @export
setMethod("*", signature(e1="lgRMatrix", e2="sparseMatrix"), multiply_csr_by_csr)

multiply_csr_by_dense = function(e1, e2) {
  if (nrow(e1) != nrow(e2) || ncol(e1) != ncol(e2))
    stop("Matrices must have the same dimensions in order to multiply them.")

  e1 = as.csr.matrix(e1)
  if (typeof(e2) == "double") {
    res = multiply_csr_by_dense_elemwise_double(e1@p, e1@j, e1@x, e2)
  } else if (typeof(e2) == "integer") {
    res = multiply_csr_by_dense_elemwise_int(e1@p, e1@j, e1@x, e2)
  } else if (typeof(e2) == "logical") {
    res = multiply_csr_by_dense_elemwise_bool(e1@p, e1@j, e1@x, e2)
  } else {
    mode(e2) = "double"
    return(e1 * e2)
  }

  out = e1
  out@x = res
  return(out)
}

#' @rdname operators
#' @export
setMethod("*", signature(e1="RsparseMatrix", e2="matrix"), multiply_csr_by_dense)

#' @rdname operators
#' @export
setMethod("*", signature(e1="ngRMatrix", e2="matrix"), multiply_csr_by_dense)

#' @rdname operators
#' @export
setMethod("*", signature(e1="lgRMatrix", e2="matrix"), multiply_csr_by_dense)

add_csr_matrices = function(e1, e2, is_substraction=FALSE) {
  if (nrow(e1) != nrow(e2) || ncol(e1) != ncol(e2))
    stop("Matrices must have the same dimensions in order to add/substract them.")

  if (inherits(e1, "nsparseMatrix") && inherits(e2, "nsparseMatrix")) {
    if (is_same_ngRMatrix(e1@p, e2@p, e1@j, e2@j)) {
      if (!is_substraction) {
        return(e1)
      } else {
        out = new("ngRMatrix")
        out@p = integer(length(e1@p))
        out@Dim = e1@Dim
        out@Dimnames = e1@Dimnames
        return(out)
      }
    }
  }
  e1 = as.csr.matrix(e1)
  e2 = as.csr.matrix(e2)
  res = add_csr_elemwise(e1@p, e2@p, e1@j, e2@j, e1@x, e2@x, is_substraction)
  out = new("dgRMatrix")
  out@Dim = e1@Dim
  out@Dimnames = e1@Dimnames
  out@p = res$indptr
  out@j = res$indices
  out@x = res$values
  return(out)
}

#' @rdname operators
#' @export
setMethod("+", signature(e1="RsparseMatrix", e2="sparseMatrix"), function(e1, e2) {
  return(add_csr_matrices(e1, e2, FALSE))
})

#' @rdname operators
#' @export
setMethod("+", signature(e1="ngRMatrix", e2="sparseMatrix"), function(e1, e2) {
  return(add_csr_matrices(e1, e2, FALSE))
})

#' @rdname operators
#' @export
setMethod("+", signature(e1="lgRMatrix", e2="sparseMatrix"), function(e1, e2) {
  return(add_csr_matrices(e1, e2, FALSE))
})


#' @rdname operators
#' @export
setMethod("-", signature(e1="RsparseMatrix", e2="sparseMatrix"), function(e1, e2) {
  return(add_csr_matrices(e1, e2, TRUE))
})

#' @rdname operators
#' @export
setMethod("-", signature(e1="ngRMatrix", e2="sparseMatrix"), function(e1, e2) {
  return(add_csr_matrices(e1, e2, TRUE))
})

#' @rdname operators
#' @export
setMethod("-", signature(e1="lgRMatrix", e2="sparseMatrix"), function(e1, e2) {
  return(add_csr_matrices(e1, e2, TRUE))
})


# nocov start

#' @rdname operators
#' @export
setMethod("*", signature(e1="RsparseMatrix", e2="integer"), function(e1, e2) {
  if (NROW(e2) != 1L) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 * e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x * as.numeric(e2)
  return(e1)
})

#' @rdname operators
#' @export
setMethod("*", signature(e1="RsparseMatrix", e2="numeric"), function(e1, e2) {
  if (NROW(e2) != 1L) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 * e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x * e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("*", signature(e1="RsparseMatrix", e2="logical"), function(e1, e2) {
  if (NROW(e2) != 1L) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 * e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x * e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("/", signature(e1="RsparseMatrix", e2="integer"), function(e1, e2) {
  if (NROW(e2) != 1L || e2 == 0) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 / e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x / as.numeric(e2)
  return(e1)
})

#' @rdname operators
#' @export
setMethod("/", signature(e1="RsparseMatrix", e2="numeric"), function(e1, e2) {
  if (NROW(e2) != 1L || e2 == 0) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 / e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x / e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("/", signature(e1="RsparseMatrix", e2="logical"), function(e1, e2) {
  if (NROW(e2) != 1L || e2 == 0) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 / e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x / e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("%/%", signature(e1="RsparseMatrix", e2="integer"), function(e1, e2) {
  if (NROW(e2) != 1L || e2 == 0) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 %/% e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x %/% as.numeric(e2)
  return(e1)
})

#' @rdname operators
#' @export
setMethod("%/%", signature(e1="RsparseMatrix", e2="numeric"), function(e1, e2) {
  if (NROW(e2) != 1L || e2 == 0) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 %/% e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x %/% e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("%/%", signature(e1="RsparseMatrix", e2="logical"), function(e1, e2) {
  if (NROW(e2) != 1L || e2 == 0) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 %/% e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x %/% e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("%%", signature(e1="RsparseMatrix", e2="integer"), function(e1, e2) {
  if (NROW(e2) != 1L || e2 == 0) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 %% e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x %% as.numeric(e2)
  return(e1)
})

#' @rdname operators
#' @export
setMethod("%%", signature(e1="RsparseMatrix", e2="numeric"), function(e1, e2) {
  if (NROW(e2) != 1L || e2 == 0) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 %% e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x %% e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("%%", signature(e1="RsparseMatrix", e2="logical"), function(e1, e2) {
  if (NROW(e2) != 1L || e2 == 0) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 %% e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x %% e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("^", signature(e1="RsparseMatrix", e2="integer"), function(e1, e2) {
  if (NROW(e2) != 1L) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 ^ e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x ^ as.numeric(e2)
  return(e1)
})

#' @rdname operators
#' @export
setMethod("^", signature(e1="RsparseMatrix", e2="numeric"), function(e1, e2) {
  if (NROW(e2) != 1L) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 ^ e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x ^ e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("^", signature(e1="RsparseMatrix", e2="logical"), function(e1, e2) {
  if (NROW(e2) != 1L) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 ^ e2)
  }
  e1 = as.csr.matrix(e1)
  e1@x = e1@x ^ e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("&", signature(e1="RsparseMatrix", e2="logical"), function(e1, e2) {
  if (NROW(e2) != 1L || !isTRUE(e2)) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 & e2)
  }
  e1 = as.csr.matrix(e1, logical=TRUE)
  e1@x = e1@x & e2
  return(e1)
})

#' @rdname operators
#' @export
setMethod("|", signature(e1="RsparseMatrix", e2="logical"), function(e1, e2) {
  if (NROW(e2) != 1L || typeof(e2) != "logical" || !isFALSE(e2)) {
    e1 = as(e1, "CsparseMatrix")
    return(e1 | e2)
  }
  e1 = as.csr.matrix(e1, logical=TRUE)
  e1@x = e1@x | e2
  return(e1)
})

#' @name mathematical-functions
#' @title Mathematical functions for CSR matrices
#' @description Implements some mathematical functions for CSR matrices
#' (a.k.a. "RsparseMatrix") without converting them to CSC matrices in the process.
#'
#' These functions reduce to applying the same function over the non-zero elements only,
#' and as such do not benefit from any storage format conversion as done implicitly
#' in the `Matrix` package.
#' @param x A CSR matrix.
#' @param digits See \link{round} and \link{signif}. If passing more than one value,
#' will call the corresponding function from the `Matrix` package, which implies first
#' converting `x` to CSC format.
#' @return A CSR matrix in `dgRMatrix` format.
#' @examples
#' library(Matrix)
#' library(rsparse)
#' set.seed(1)
#' X = as.csr.matrix(rsparsematrix(4, 3, .4))
#' abs(X)
#' sqrt(X^2)
NULL

#' @rdname mathematical-functions
#' @export
setMethod("sqrt", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = sqrt(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("abs", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = abs(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("log1p", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = log1p(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("cos", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = cos(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("tan", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = tan(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("tanh", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = tanh(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("tanpi", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = tanpi(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("sinh", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = sinh(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("atanh", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = atanh(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("expm1", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = expm1(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("sign", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = sign(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("ceiling", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = ceiling(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("floor", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = floor(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("trunc", signature(x="RsparseMatrix"), function(x) {
  x = as.csr.matrix(x)
  x@x = trunc(x@x)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("round", signature(x="RsparseMatrix", digits="ANY"), function(x, digits) {
  if (!missing(digits) && NROW(digits) != 1L) {
    x = as(x, "CsparseMatrix")
    return(round(x, digits))
  }
  x = as.csr.matrix(x)
  x@x = round(x@x, digits)
  return(x)
})

#' @rdname mathematical-functions
#' @export
setMethod("signif", signature(x="RsparseMatrix", digits="ANY"), function(x, digits) {
  if (!missing(digits) && NROW(digits) != 1L) {
    x = as(x, "CsparseMatrix")
    return(signif(x, digits))
  }
  x = as.csr.matrix(x)
  x@x = signif(x@x, digits)
  return(x)
})

# nocov end

get_indices_integer = function(i, max_i, index_names) {

  if (inherits(i, "nsparseVector")) {
    if (i@length > max_i) {
      stop("Dimension of indexing vector is larger than matrix to subset.")
    } else if (i@length == max_i) {
      i = i@i
    } else { ### mimic of R base's recycling
      full_repeats = max_i %/% i@length
      remainder = max_i - i@length*full_repeats
      i = repeat_indices_n_times(i@i, i[seq(1L, remainder)]@i, i@length, max_i)
    }
    if (typeof(i) != "integer")
      i = as.integer(i)
  }

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

get_x_values = function(Mat) {
  if (.hasSlot(Mat, "x")) {
    return(as.numeric(Mat@x))
  } else {
    return(numeric())
  }
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

  if (!inherits(x, c("dgRMatrix", "ngRMatrix", "lgRMatrix")))
    x = as.csr.matrix(x)
  has_x = !inherits(x, "ngRMatrix")

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
    res = new(class(x))
    res@p = integer(NROW(i) + 1L)
    res@Dim = c(NROW(i), NROW(j))

    row_names = if(is.null(row_names) || !NROW(row_names)) NULL else row_names[i]
    col_names = if(is.null(col_names) || !NROW(col_names)) NULL else col_names[j]
    res@Dimnames = list(row_names, col_names)
    return(res)
  }

  if (all_i && all_j) {
    return(x)
  } else if (length(x@j) == 0L) {
    indptr = integer(n_row + 1)
    col_indices = integer()
    x_values = numeric()
  } else if (i_is_seq && all_j) {
    first = x@p[i[1L]] + 1L
    last = x@p[i[n_row] + 1L]
    indptr = x@p[seq(i[1L], i[n_row]+1L)] - x@p[i[1L]]
    col_indices = x@j[first:last]
    if (has_x)
      x_values = x@x[first:last]
  } else if (!i_is_seq && all_j) {
    temp = copy_csr_rows(x@p, x@j, get_x_values(x), i-1L)
    indptr = temp$indptr
    col_indices = temp$indices
    if (has_x)
      x_values = temp$values
  } else if (j_is_seq) {
    temp = copy_csr_rows_col_seq(x@p, x@j, get_x_values(x), i-1L, j-1L)
    indptr = temp$indptr
    col_indices = temp$indices
    if (has_x)
      x_values = temp$values
  } else {
    temp = copy_csr_arbitrary(x@p, x@j, get_x_values(x), i-1L, j-1L)
    indptr = temp$indptr
    col_indices = temp$indices
    if (has_x)
      x_values = temp$values
  }

  res = new(class(x)[1L])
  res@p = indptr
  res@j = col_indices
  if (has_x) {
    if (inherits(x, "lgRMatrix")) {
        res@x = as.logical(x_values)
      } else {
        res@x = x_values
      }
  }
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

#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "nsparseVector", j = "nsparseVector", drop="logical"), subset_csr)
#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "missing", j = "nsparseVector", drop="logical"), subset_csr)
#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "nsparseVector", j = "missing", drop="logical"), subset_csr)

#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "nsparseVector", j = "nsparseVector", drop="missing"), subset_csr)
#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "missing", j = "nsparseVector", drop="missing"), subset_csr)
#' @rdname slice
#' @export
setMethod(`[`, signature(x = "RsparseMatrix", i = "nsparseVector", j = "missing", drop="missing"), subset_csr)


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

#' @name casting
#' @title Conversions between matrix types
#' @description Convenience functions for converting to different sparse matrix formats,
#' between pairs of classes which might not be supported in the `Matrix` package.
#'
#' These come in the form of explicit functions 'as.<type>.matrix' (see below),
#' as well as registered conversion methods to use along with `as(object, type)`, adding
#' extra conversion routes which are missing in the `Matrix` package for output
#' types `dgRMatrix`, `lgRMatrix`, and `ngRMatrix`.
#' @details The functions internally use as(x, "?sparseMatrix"), so they might work
#' with other object classes if they register a conversion method for `Matrix` base
#' types.
#'
#' When passed a vector, the functions `as.csr.matrix` and `as.coo.matrix` will
#' assume that it is a row vector, while `as.csc.matrix` will assume it's a column vector.
#' @param x A matrix which is to be converted to a different format.
#'
#' Supported input types are:\itemize{
#' \item Sparse matrices from `Matrix` package, in any format.
#' \item Sparse vectors from `Matrix` in any format.
#' \item Dense matrices from base R (class `matrix`).
#' \item Dense vectors from base R (classes `numeric`, `integer`, `logical`).
#' \item Dense matrix or vector from package `float` (class `float32`).
#' \item `data.frame`, `data.table`, and `tibble`.
#' }
#' @param binary Whether the result should be a binary-only matrix (inheriting from
#' class `nsparseMatrix` - these don't have slot `x`).
#' Can only pass one of `binary` or `logical`.
#' @param logical Whether the result should be a matrix with logical (boolean) type
#' (inheriting from `lsparseMatrix`).
#' Can only pass one of `binary` or `logical`.
#' @return A sparse matrix, with format:\itemize{
#' \item CSR (a.k.a. `RsparseMatrix`) when calling `as.csr.matrix`
#' (class `dgRMatrix`, `ngRMatrix`, or `lgRMatrix`, depending on parameters `binary` and `logical`).
#' \item CSC (a.k.a. `CsparseMatrix`) when calling `as.csc.matrix`
#' (class `dgCMatrix`, `ngCMatrix`, or `lgCMatrix`, depending on parameters `binary` and `logical`).
#' \item COO (a.k.a. `TsparseMatrix`) when calling `as.coo.matrix`
#' (class `dgTMatrix`, `ngTMatrix`, or `lgTMatrix`, depending on parameters `binary` and `logical`).
#' }
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
#'
#' ### Using the new conversion methods
#' ### (these would fail if 'rsparse' is not loaded)
#' as(matrix(1:3), "ngRMatrix")
#' as(as.csc.matrix(m.coo), "dgRMatrix")
NULL

#' @rdname casting
#' @export
as.csr.matrix = function(x, binary=FALSE, logical=FALSE) {
  if (binary && logical)
    stop("Can pass only one of 'binary' or 'logical'.")

  if ((inherits(x, "dgRMatrix") && !binary && !logical) ||
      (inherits(x, "ngRMatrix") && binary) ||
      (inherits(x, "lgRMatrix") && logical)) {
    return(x)
  }

  if (inherits(x, "float32"))
    x = float::dbl(x)

  if (inherits(x, c("numeric", "integer", "logical")))
    x = matrix(x, nrow=1L)

  if (inherits(x, c("data.frame", "tibble", "data.table")))
    x = as.matrix(x)

  if (!binary && !logical) {
    target_class = "dgRMatrix"
  } else if (binary) {
    target_class = "ngRMatrix"
  } else {
    target_class = "lgRMatrix"
  }

  if (inherits(x, "sparseVector")) {
    X.csr = new(target_class)
    X.csr@Dim = c(1L, x@length)
    X.csr@p = c(0L, length(x@i))
    X.csr@j = x@i - 1L
    if (!binary) {
      if (inherits(x, "dsparseVector")) {
        if (!logical)
          X.csr@x = x@x
        else
          X.csr@x = as.logical(x@x)
      } else if (inherits(x, "isparseVector")) {
        if (!logical)
          X.csr@x = as.numeric(x@x)
        else
          X.csr@x = as.logical(x@x)
      } else if (inherits(x, "lsparseVector")) {
          if (!logical)
            X.csr@x = as.numeric(x@x)
          else
            X.csr@x = x@x
      } else {
        if (!logical)
          X.csr@x = rep(1., length(x@i))
        else
          X.csr@x = rep(TRUE, length(x@i))
      }
    }
    x = X.csr
  }

  if (!inherits(x, "RsparseMatrix"))
    x = as(x, "RsparseMatrix")

  if (!binary && !logical && !inherits(x, "dgRMatrix")) {
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

  if (logical && !inherits(x, "lgRMatrix")) {
    X.csr = new("lgRMatrix")
    X.csr@Dim = x@Dim
    X.csr@Dimnames = x@Dimnames
    X.csr@p = x@p
    X.csr@j = x@j
    if (.hasSlot(x, "x"))
      X.csr@x = as.logical(x@x)
    else
      X.csr@x = rep(TRUE, length(x@j))
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
as.csc.matrix = function(x, binary=FALSE, logical=FALSE) {
  if (binary && logical)
    stop("Can pass only one of 'binary' or 'logical'.")

  if ((inherits(x, "dgCMatrix") && !binary && !logical) ||
      (inherits(x, "ngCMatrix") && binary) ||
      (inherits(x, "lgCMatrix") && logical)) {
    return(x)
  }

  if (inherits(x, "float32"))
    x = float::dbl(x)

  if (inherits(x, c("numeric", "integer", "logical", "data.frame", "tibble", "data.table")))
    x = as.matrix(x)

  if (!inherits(x, "CsparseMatrix"))
    x = as(x, "CsparseMatrix")

  if (!binary && !logical && !inherits(x, "dgCMatrix")) {
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

  if (logical && !inherits(x, "lgCMatrix")) {
    X.csc = new("dgCMatrix")
    X.csc@Dim = x@Dim
    X.csc@Dimnames = x@Dimnames
    X.csc@p = x@p
    X.csc@i = x@i
    if (.hasSlot(x, "x"))
      X.csc@x = as.logical(x@x)
    else
      X.csc@x = rep(TRUE, length(x@i))
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
as.coo.matrix = function(x, binary=FALSE, logical=FALSE) {
  if (binary && logical)
    stop("Can pass only one of 'binary' or 'logical'.")

  if ((inherits(x, "dgTMatrix") && !binary && !logical) ||
      (inherits(x, "ngTMatrix") && binary) ||
      (inherits(x, "lgTMatrix") && logical)) {
    return(x)
  }

  if (inherits(x, "float32"))
    x = float::dbl(x)

  if (inherits(x, c("numeric", "integer", "logical")))
    x = matrix(x, nrow=1L)

  if (inherits(x, c("data.frame", "tibble", "data.table")))
    x = as.matrix(x)

  if (inherits(x, "sparseVector"))
    x = as.csr.matrix(x)

  if (!inherits(x, "TsparseMatrix"))
    x = as(x, "TsparseMatrix")

  if (!binary && !logical && !inherits(x, "dgTMatrix")) {
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

  if (logical && !inherits(x, "dgTMatrix")) {
    X.coo = new("lgTMatrix")
    X.coo@Dim = x@Dim
    X.coo@Dimnames = x@Dimnames
    X.coo@i = x@i
    X.coo@j = x@j
    if (.hasSlot(x, "x"))
      X.coo@x = as.logical(x@x)
    else
      X.coo@x = rep(TRUE, length(x@j))
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

#' @export
setAs("dgCMatrix", "dgRMatrix", function(from) as.csr.matrix(from))
#' @export
setAs("ngCMatrix", "dgRMatrix", function(from) as.csr.matrix(from))
#' @export
setAs("lgCMatrix", "dgRMatrix", function(from) as.csr.matrix(from))
#' @export
setAs("dgTMatrix", "dgRMatrix", function(from) as.csr.matrix(from))
#' @export
setAs("ngTMatrix", "dgRMatrix", function(from) as.csr.matrix(from))
#' @export
setAs("lgTMatrix", "dgRMatrix", function(from) as.csr.matrix(from))
#' @export
setAs("ngRMatrix", "dgRMatrix", function(from) as.csr.matrix(from))
#' @export
setAs("lgRMatrix", "dgRMatrix", function(from) as.csr.matrix(from))
#' @export
setAs("sparseVector", "dgRMatrix", function(from) as.csr.matrix(from))

#' @export
setAs("dgCMatrix", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))
#' @export
setAs("ngCMatrix", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))
#' @export
setAs("lgCMatrix", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))
#' @export
setAs("dgTMatrix", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))
#' @export
setAs("ngTMatrix", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))
#' @export
setAs("lgTMatrix", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))
#' @export
setAs("ngRMatrix", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))
#' @export
setAs("dgRMatrix", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))
#' @export
setAs("sparseVector", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))

#' @export
setAs("dgCMatrix", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))
#' @export
setAs("ngCMatrix", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))
#' @export
setAs("lgCMatrix", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))
#' @export
setAs("dgTMatrix", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))
#' @export
setAs("ngTMatrix", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))
#' @export
setAs("lgTMatrix", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))
#' @export
setAs("lgRMatrix", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))
#' @export
setAs("dgRMatrix", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))
#' @export
setAs("sparseVector", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))

#' @export
setAs("matrix", "dgRMatrix", function(from) as.csr.matrix(from))
#' @export
setAs("matrix", "lgRMatrix", function(from) as.csr.matrix(from, logical=TRUE))
#' @export
setAs("matrix", "ngRMatrix", function(from) as.csr.matrix(from, binary=TRUE))


t_csc_to_csr = function(X) {
  out = new(gsub("CMatrix", "RMatrix", class(X)[1L], ignore.case=FALSE))
  out@Dim = rev(X@Dim)
  out@Dimnames = rev(X@Dimnames)
  out@p = X@p
  out@j = X@i
  if (!inherits(X, "ngCMatrix"))
    out@x = X@x
  return(out)
}

t_csr_to_csc = function(X) {
  out = new(gsub("RMatrix", "CMatrix", class(X)[1L], ignore.case=FALSE))
  out@Dim = rev(X@Dim)
  out@Dimnames = rev(X@Dimnames)
  out@p = X@p
  out@i = X@j
  if (!inherits(X, "ngRMatrix"))
    out@x = X@x
  return(out)
}

#' @title Transpose a sparse matrix by changing its format
#' @description Transposes a sparse matrix in CSC (a.k.a. "CsparseMatrix")
#' or CSR (a.k.a. "RsparseMatrix") formats by converting it to the opposite format
#' (i.e. CSC -> CSR, CSR -> CSC).
#'
#' This implies only a shallow copy (i.e. it's faster), as the only necessary thing to make
#' such transpose operation is to swap the number of rows and columns and change the class
#' of the object (all other slots remain the same), avoiding any deep copying and
#' format conversion as when e.g. creating a CSC transpose of a CSC matrix.
#'
#' If the input is neither a CSR not CSC matrix, it will just call the generic `t()` method.
#' @param X A sparse matrix in CSC (`dgCMatrix` or `ngCMatrix`) or CSR (`dgRMatrix` or `ngRMatrix`) formats. If `X` is of a different type, will just invoke its generic
#' `t()` method.
#' @returns The transpose of `X` (rows become columns and columns become rows),
#' but in the opposite format (CSC -> CSR, CSR -> CSC).
#' @examples
#' library(Matrix)
#' library(rsparse)
#' set.seed(1)
#' X = rsparsematrix(3, 4, .5)
#' inherits(X, "CsparseMatrix")
#' Xtrans = t_shallow(X)
#' inherits(Xtrans, "RsparseMatrix")
#' nrow(X) == ncol(Xtrans)
#' ncol(X) == nrow(Xtrans)
#'
#' Xorig = t_shallow(Xtrans)
#' inherits(Xorig, "CsparseMatrix")
#' @export
t_shallow = function(X) {
  if (inherits(X, c("dgCMatrix", "ngCMatrix", "lgCMatrix"))) {
    return(t_csc_to_csr(X))
  } else if (inherits(X, c("dgRMatrix", "ngRMatrix", "lgRMatrix"))) {
    return(t_csr_to_csc(X))
  } else {
    return(t(X))
  }
}

#' @title Concatenate inputs by rows into a CSR matrix
#' @description Concatenate two or more matrices and/or vectors by rows, giving a CSR matrix
#' as result.
#'
#' This is aimed at concatenating several CSR matrices or sparse vectors at a time,
#' as it will be faster than calling `rbind` which will only concatenate one at a
#' time, resulting in unnecessary allocations.
#'
#' \bold{Important:} for the sake of speed, this function will omit any row or column names
#' that the inputs might have.
#' @param ... Inputs to concatenate. The function is aimed at CSR matrices (`dgRMatrix`,
#' `ngRMatrix`, `lgRMatrix`) and sparse vectors (`sparseVector`). It will work with other classes
#' (such as `dgCMatrix`) but will not be as efficient.
#' @returns A CSR matrix (class `dgRMatrix`, `lgRMatrix`, or `ngRMatrix` depending on the inputs) with
#' the inputs concatenated by rows.
#' @seealso \link{rbind2-method}
#' @examples
#' library(Matrix)
#' library(rsparse)
#' v = as(1:10, "sparseVector")
#' rbind_csr(v, v, v)
#'
#' X = matrix(1:20, nrow=2)
#' rbind_csr(X, v)
#' @export
rbind_csr = function(...) {

  binary_types = c("nsparseMatrix", "nsparseVector")
  logical_types = c("lsparseMatrix", "lsparseVector", "logical")
  cast_csr_same = function(x) as.csr.matrix(x, binary=inherits(x, binary_types), logical=inherits(x, logical_types))
  cast_if_not_csr = function(x) {
    if (inherits(x, c("dgRMatrix", "ngRMatrix", "lgRMatrix", "sparseVector"))) {
      return(x)
    } else {
      return(cast_csr_same(x))
    }
  }
  args = lapply(list(...), cast_if_not_csr)

  if (length(args) == 0L) {
    return(new("dgRMatrix"))
  } else if (length(args) == 1L) {
    return(cast_csr_same(args[[1L]]))
  } else if (length(args) == 2L) {
    return(rbind2(args[[1L]], args[[2L]]))
  }

  is_binary = sapply(args, function(x) inherits(x, binary_types))
  is_logical = sapply(args, function(x) inherits(x, logical_types))
  is_numeric = !is_binary & !is_logical
  if (any(is_numeric)) {
    out = new("dgRMatrix")
  } else if (any(is_logical)) {
    out = new("lgRMatrix")
  } else {
    out = new("ngRMatrix")
  }

  nrows = sum(sapply(args, function(x) ifelse(inherits(x, "sparseMatrix"), nrow(x), 1L)))
  ncols = max(sapply(args, function(x) ifelse(inherits(x, "sparseMatrix"), ncol(x), length(x))))
  nnz = sum(sapply(args, function(x) ifelse(inherits(x, "sparseMatrix"), length(x@j), length(x@i))))
  if (nrows >= .Machine$integer.max)
    stop("Result has too many rows for R to handle.")
  if (nnz >= .Machine$integer.max)
    stop("Result has too many non-zero entries for R to handle.")

  out@p = integer(nrows + 1L)
  out@j = integer(nnz)
  if (inherits(out, "dgRMatrix")) {
    out@x = numeric(nnz)
  } else if (inherits(out, "lgRMatrix")) {
    out@x = logical(nnz)
  }
  out@Dim = as.integer(c(nrows, ncols))
  out@Dimnames = list(NULL, NULL)
  if (!nrows || !ncols)
    return(out)

  out = concat_csr_batch(args, out)
  return(out)
}

#' @name rbind2-method
#' @title Concatenate CSR matrices by rows
#' @description `rbind2` method for the `dgRMatrix`, `ngRMatrix`, and `sparseVector` classes
#' from the `Matrix` package. This method will concatenate CSR (a.k.a. RsparseMatrix)
#' matrix objects without converting them to triplets/COO in the process, and will also
#' concatenate sparse vectors assuming they are row vectors.
#' @param x First matrix to concatenate.
#' @param y Second matrix to concatenate.
#' @return A CSR matrix (`dgRMatrix` or `ndRMatrix` depending on the inputs).
#' @examples
#' library(Matrix)
#' library(rsparse)
#' set.seed(1)
#' X = rsparsematrix(3, 4, .3)
#' X = as(X, "RsparseMatrix")
#' inherits(rbind2(X, X), "dgRMatrix")
#' inherits(rbind(X, X, X, X), "dgRMatrix")
NULL

concat_dimname = function(nm1, nm2, nrow1, nrow2) {
  if (is.null(nm1) && is.null(nm2)) {
    return(NULL)
  } else if (!is.null(nm1) && !is.null(nm2)) {
    return(c(nm1, nm2))
  }

  if (is.null(nm1)) {
    nm1 = as.character(seq(1, nrow1))
  }
  if (is.null(nm2)) {
    nm2 = as.character(seq(nrow1 + 1L, nrow1 + nrow2))
  }
  return(concat_dimname(nm1, nm2, nrow1, nrow2))
}

concat_dimnames = function(mat1, mat2) {
  dim1 = concat_dimname(mat1@Dimnames[[1L]], mat2@Dimnames[[1L]], nrow(mat1), nrow(mat2))
  dim2 = concat_dimname(mat1@Dimnames[[2L]], mat2@Dimnames[[2L]], ncol(mat1), ncol(mat2))
  return(list(dim1, dim2))
}

concat_as_numeric = function(v1, v2) {
  if (typeof(v1) != "double")
    v1 = as.numeric(v1)
  if (typeof(v2) != "double")
    v2 = as.numeric(v2)
  return(c(v1, v2))
}

concat_as_logical = function(v1, v2) {
  if (typeof(v1) != "logical")
    v1 = as.logical(v1)
  if (typeof(v2) != "logical")
    v2 = as.logical(v2)
  return(c(v1, v2))
}

rbind2_csr = function(x, y, out) {
  out@Dim = c(x@Dim[1L] + y@Dim[1L], max(x@Dim[2L], y@Dim[2L]))
  if (out@Dim[2L] >= .Machine$integer.max)
    stop("Resulting matrix has too many rows for R to handle.")
  out@Dimnames = concat_dimnames(x, y)
  out@p = concat_indptr2(x@p, y@p)
  out@j = c(x@j, y@j)
  return(out)
}

rbind2_dgr = function(x, y) {
  out = new("dgRMatrix")
  out = rbind2_csr(x, y, out)
  if (.hasSlot(x, "x") && .hasSlot(y, "x")) {
    out@x = concat_as_numeric(x@x, y@x)
  } else if (.hasSlot(x, "x")) {
    out@x = concat_as_numeric(x@x, rep(1., nrow(y)))
  } else if (.hasSlot(y, "x")) {
    out@x = concat_as_numeric(rep(1., nrow(x)), y@x)
  } else {
    out@x = rep(1., nrow(x) + nrow(y))
  }
  return(out)
}

rbind2_lgr = function(x, y) {
  out = new("lgRMatrix")
  out = rbind2_csr(x, y, out)
  if (.hasSlot(x, "x") && .hasSlot(y, "x")) {
    out@x = concat_as_logical(x@x, y@x)
  } else if (.hasSlot(x, "x")) {
    out@x = concat_as_logical(x@x, rep(TRUE, nrow(y)))
  } else if (.hasSlot(y, "x")) {
    out@x = concat_as_logical(rep(TRUE, nrow(x)), y@x)
  } else {
    out@x = rep(TRUE, nrow(x) + nrow(y))
  }
  return(out)
}

rbind2_ngr = function(x, y) {
  out = new("ngRMatrix")
  out = rbind2_csr(x, y, out)
  return(out)
}

rbind2_generic = function(x, y) {
  binary_types = c("nsparseMatrix", "nsparseVector")
  logical_types = c("lsparseMatrix", "lsparseVector")
  x_is_binary = inherits(x, binary_types)
  x_is_logical = inherits(x, logical_types)
  y_is_binary = inherits(y, binary_types)
  y_is_logical = inherits(y, logical_types)

  if (x_is_binary && y_is_binary) {
    return(rbind2_ngr(as.csr.matrix(x, binary=TRUE), as.csr.matrix(y, binary=TRUE)))
  } else if ((x_is_binary || x_is_logical) && (y_is_binary || y_is_logical)) {
    return(rbind2_lgr(as.csr.matrix(x, logical=TRUE), as.csr.matrix(y, logical=TRUE)))
  } else {
    return(rbind2_dgr(as.csr.matrix(x), as.csr.matrix(y)))
  }
}

#' @rdname rbind2-method
#' @export
setMethod("rbind2", signature(x="RsparseMatrix", y="RsparseMatrix"), rbind2_generic)

#' @rdname rbind2-method
#' @export
setMethod("rbind2", signature(x="sparseVector", y="RsparseMatrix"), rbind2_generic)

#' @rdname rbind2-method
#' @export
setMethod("rbind2", signature(x="RsparseMatrix", y="sparseVector"), rbind2_generic)

#' @rdname rbind2-method
#' @export
setMethod("rbind2", signature(x="sparseVector", y="sparseVector"), rbind2_generic)
