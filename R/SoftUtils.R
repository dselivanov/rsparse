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

pad_svd = function(x, rank) {
  stopifnot(length(x$d) <= rank)
  nr = nrow(x$u)
  nc = nrow(x$v)

  x_rank = length(x$d)
  x_rank_true = sum(x$d > 0)
  n_pad = rank - x_rank
  if(n_pad > 0) {
    x$d = c(x$d, rep(x$d[x_rank], n_pad) )

    u_pad = matrix(rnorm(n_pad * nr), nr, n_pad)
    u_pad = u_pad - x$u %*% (t(x$u) %*% u_pad)
    u_pad = qr.Q(qr(u_pad, LAPACK = TRUE))
    x$u = cbind(x$u, u_pad); rm(u_pad)

    v_pad = matrix(rnorm(n_pad * nc), nc, n_pad)
    v_pad = v_pad - x$v %*% crossprod(x$v, v_pad)
    v_pad = qr.Q(qr(v_pad, LAPACK = TRUE))
    x$v = cbind(x$v, v_pad)
    x
  } else {
    x
  }
}
