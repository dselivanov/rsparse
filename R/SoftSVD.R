# implements Rank-Restricted Soft SVD
# algorithm 2.1 from https://arxiv.org/pdf/1410.2596.pdf

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
  trace_iter = vector("list", n_iter)

  k = 1L
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

    trace_iter[[k]] = list(iter = i, scorer = "frob_delta", value = frob_delta)
    k = k + 1L

    futile.logger::flog.debug("reco:::soft_svd: iter %d, frobenious norm delta %.3f", i, frob_delta)
    svd_old = svd_new
    if(frob_delta < convergence_tol) {
      futile.logger::flog.debug("reco:::soft_svd: converged with tol %f after %d iter", convergence_tol, i)
      CONVERGED = TRUE
      break
    }
  }
  data.table::setattr(svd_new, "trace", data.table::rbindlist(trace_iter))
  if(!CONVERGED)
    futile.logger::flog.warn("reco:::soft_svd: didn't converged with tol %f after %d iterations - returning latest solution", convergence_tol, i)
  svd_new
}

# workhorse for soft_svd
solve_iter_als_svd = function(x, svd_current, lambda, singular_vectors = c("u", "v")) {
  singular_vectors = match.arg(singular_vectors)
  m = (x %*% svd_current[[singular_vectors]]) %*% diag((svd_current$d / (svd_current$d + lambda)))
  svd_econ(m)
}
