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

solve_iter_als_softimpute = function(x, svd_current, lambda, singular_vectors = c("u", "v")) {
  singular_vectors = match.arg(singular_vectors)

  if(singular_vectors == "v") {
    A = t(svd_current$u) * sqrt(svd_current$d)
    B = t(svd_current$v) * sqrt(svd_current$d)
  } else {
    A = t(svd_current$v) * sqrt(svd_current$d)
    B = t(svd_current$u) * sqrt(svd_current$d)
  }

  x_delta = x
  # make_sparse_approximation calculates values of sparse matrix X_new = X - A %*% B
  # for only non-zero values of X
  upd = make_sparse_approximation(x, A, B)
  x_delta@x = x@x - upd
  flog.debug("soft_impute_als objective %.5f", (as.numeric(crossprod(x_delta@x)) + lambda * sum(svd_current$d)) / length(x_delta@x))

  first = (x_delta %*% svd_current[[singular_vectors]]) %*% diag( sqrt(svd_current$d) / (svd_current$d + lambda))
  second = t(A * (svd_current$d / (svd_current$d + lambda)))

  m = first + second
  reco:::svd_econ(m %*% diag(sqrt(svd_current$d)))
}

soft_impute = function(x, rank = 10L, lambda = 0, n_iter = 10L, convergence_tol = 1e-3, init = NULL) {
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
    # Bsvd = solve_iter_als_softimpute(tx, t(B), t(A), svd_new, lambda, "u")
    Bsvd = solve_iter_als_softimpute(tx, svd_new, lambda, "u")
    # str(Bsvd)
    svd_new$v = Bsvd$u
    svd_new$d = Bsvd$d
    # not sure why this line is required
    svd_new$u = svd_new$u %*% Bsvd$v

    Asvd = solve_iter_als_softimpute(x, svd_new, lambda, "v")
    # str(Asvd)

    svd_new$u = Asvd$u
    svd_new$d = Asvd$d
    # not sure why this line is required
    svd_new$v = svd_new$v %*% Asvd$v

    frob_delta = calc_frobenius_norm_delta(svd_old, svd_new)
    futile.logger::flog.debug("soft_impute: iter %d, delta frobenious norm %.5f", i, frob_delta)
    svd_old = svd_new
    if(frob_delta < convergence_tol) {
      futile.logger::flog.debug("soft_impute: converged with tol %f after %d iter", convergence_tol, i)
      CONVERGED = TRUE
      break
    }
  }
  if(!CONVERGED)
    futile.logger::flog.warn("soft_impute: didn't converged with tol %f after %d iterations - returning latest solution", convergence_tol, i)
  svd_new
}
