# implements Rank-Restricted Soft SVD
# algorithm 2.1 from https://arxiv.org/pdf/1410.2596.pdf
solve_iter_als_svd = function(xx, svd_current, lambda, mult = c("u", "v")) {
  mult = match.arg(mult)
  tmp = (xx %*% svd_current[[mult]]) %*% diag((svd_current$d / (svd_current$d + lambda)))
  is(!is.matrix(tmp))
    tmp = as.matrix(tmp)
  svd_econ(tmp)
}

soft_svd = function(x, rank = 10L, lambda = 0, n_iter = 10L, convergence_tol = 1e-3, init = NULL) {
  tx = t(x)
  if(is.null(init)) {
    # draw random matrix and make columns orthogonal with QR decomposition
    U = matrix(rnorm(n = nrow(x) * rank), nrow = nrow(x))
    U = qr.Q(qr(U, LAPACK = TRUE))
    # init with dummy values
    D  = rep(1, rank)
    V = matrix(rep(0, ncol(x) * rank), nrow = ncol(x))
    svd_old = list(d = D, u = U, v = V); rm(U, V)
  } else {
    # warm start with another SVD
    stopifnot(is.list(init))
    stopifnot(all(names(init) %in% c("u", "d", "v")))
    stopifnot(isTRUE(all.equal(dim(init$u), c(nrow(x), rank))))
    stopifnot(isTRUE(all.equal(dim(init$v), c(ncol(x), rank))))
    stopifnot(length(init$d) == rank)
    svd_old = init
  }

  svd_new = svd_old
  CONVERGED = FALSE
  for(i in seq_len(n_iter)) {
    Bsvd = solve_iter_als_svd(tx, svd_new, lambda, "u")
    svd_new$v = Bsvd$u
    svd_new$d = Bsvd$d
    Asvd = solve_iter_als_svd(x, svd_new, lambda, "v")
    svd_new$u = Asvd$u
    svd_new$d = Asvd$d
    # not sure about this line - found in reference implementation
    # https://github.com/cran/softImpute/blob/a5c6e4bd5a660d6a79119991b0cbd4923dbe9b66/R/Ssvd.als.R#L64
    svd_new$v = svd_new$v %*% Asvd$v
    frob_delta = calc_frobenius_norm_delta(svd_old, svd_new)

    futile.logger::flog.debug("soft_svd: iter %d, delta frobenious norm %.5f", i, frob_delta)
    svd_old = svd_new
    if(frob_delta < convergence_tol) {
      futile.logger::flog.debug("soft_svd: converged with tol %f after %d iter", convergence_tol, i)
      CONVERGED = TRUE
      break
    }
  }
  if(!CONVERGED)
    futile.logger::flog.warn("soft_svd: didn't converged with tol %f after %d iterations - returning latest solution", convergence_tol, i)
  svd_new
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
  stopifnot(is.matrix(x))
  stopifnot(is.numeric(x))
  arma_svd_econ(x)
}
