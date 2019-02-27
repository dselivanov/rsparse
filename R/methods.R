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
NULL

#' @rdname matmult
#' @export
setMethod("%*%", signature(x="dgRMatrix", y="matrix"), function(x, y) {
  if(ncol(x) != nrow(y)) stop("non-conformable arguments")
  csr_dense_tcrossprod(x, t(y), getOption("rsparse_omp_threads", parallel::detectCores()))
})

#' @rdname matmult
#' @export
setMethod("tcrossprod", signature(x="dgRMatrix", y="matrix"), function(x, y) {
  if(ncol(x) != ncol(y)) stop("non-conformable arguments")
  csr_dense_tcrossprod(x, y, getOption("rsparse_omp_threads", parallel::detectCores()))
})

#' @rdname matmult
#' @export
setMethod("%*%", signature(x="matrix", y="dgCMatrix"), function(x, y) {
  if(ncol(x) != nrow(y)) stop("non-conformable arguments")
  dense_csc_prod(x, y, getOption("rsparse_omp_threads", parallel::detectCores()))
})

#' @rdname matmult
#' @export
setMethod("crossprod", signature(x="matrix", y="dgCMatrix"), function(x, y) {
  if(nrow(x) != nrow(y)) stop("non-conformable arguments")
  dense_csc_prod(t(x), y, getOption("rsparse_omp_threads", parallel::detectCores()))
})
