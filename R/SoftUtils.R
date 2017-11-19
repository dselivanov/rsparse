make_sparse_approximation = function(x, A, B, n_threads = parallel::detectCores()) {
  stopifnot(nrow(x) == ncol(A))
  stopifnot(ncol(x) == ncol(B))
  UseMethod("make_sparse_approximation")
}

make_sparse_approximation.CsparseMatrix = function(x, A, B, n_threads = parallel::detectCores()) {
  CSC = 1L
  CSR = 2L
  cpp_make_sparse_approximation(x, A, B, CSC, n_threads)
}

make_sparse_approximation.RsparseMatrix = function(x, A, B, n_threads = parallel::detectCores()) {
  CSC = 1L
  CSR = 2L
  cpp_make_sparse_approximation(x, A, B, CSR, n_threads)
}

calc_frobenius_norm_delta = function(svd_old, svd_new) {
  denom = sum(svd_old$d ^ 2)
  utu = svd_new$d * (t(svd_new$u) %*% svd_old$u)
  vtv = svd_old$d * (t(svd_old$v) %*% svd_new$v)
  uvprod = sum(diag(utu %*% vtv))
  num = denom + sum(svd_new$d ^ 2) - 2 * uvprod
  num / max(denom, 1e-09)
}

svd_econ = function(x) {
  if(inherits(x, "denseMatrix")) x = as.matrix(x)
  stopifnot(is.matrix(x))
  stopifnot(is.numeric(x))
  arma_svd_econ(x)
}
