#' matmult
#'
#' Multithreaded Sparse-Dense Matrix Multiplication
#'
#' @details
#'
#' Accelerates sparse-dense matrix multiplications using openmp. Applicable to
#' (\code{dgRMatrix}, \code{matrix}), (\code{matrix}, \code{dgRMatrix}),
#' (\code{dgCMatrix}, \code{matrix}), (\code{matrix}, \code{dgCMatrix}) combinations
#'
#' @param x,y
#' Numeric/float matrices.
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
