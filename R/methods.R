#' @export
setMethod("%*%", signature(x="dgRMatrix", y="matrix"), function(x, y) {
  if(ncol(x) != nrow(y)) stop("non-conformable arguments")
  csr_dense_tcrossprod(x, t(y), getOption("rsparse_omp_threads", parallel::detectCores()))
})

#' @export
setMethod("tcrossprod", signature(x="dgRMatrix", y="matrix"), function(x, y) {
  if(ncol(x) != ncol(y)) stop("non-conformable arguments")
  csr_dense_tcrossprod(x, y, getOption("rsparse_omp_threads", parallel::detectCores()))
})
