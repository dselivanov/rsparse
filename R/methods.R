#' @export
setMethod("%*%", signature(x="dgRMatrix", y="matrix"), function(x, y) {
  prod_csr_dense(x, y)
})

#' @export
setMethod("crossprod", signature(x="dgRMatrix", y="matrix"), function(x, y) {
  crossprod_csr_dense(x, y)
})

#' @export
setMethod("tcrossprod", signature(x="dgRMatrix", y="matrix"), function(x, y) {
  prod_csr_dense(x, t(y))
})

#' @export
setMethod("%*%", signature(x="matrix", y="dgRMatrix"), function(x, y) {
  prod_dense_csr(x, y)
})

#' @export
setMethod("crossprod", signature(x="matrix", y="dgRMatrix"), function(x, y) {
  prod_dense_csr(t(x), y)
})

#' @export
setMethod("tcrossprod", signature(x="matrix", y="dgRMatrix"), function(x, y) {
  tcrossprod_dense_csr(x, y)
})
