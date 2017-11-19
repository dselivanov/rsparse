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

  trace_iter = vector("list", n_iter)
  k = 1L

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
    loss =  attr(Asvd, "loss")

    trace_iter[[k]] = list(iter = i, scorer = "frob_delta", value = frob_delta)
    k = k + 1L
    trace_iter[[k]] = list(iter = i, scorer = "loss", value = loss)
    k = k + 1L

    futile.logger::flog.debug("reco:::soft_impute: iter %03d, loss %.3f frobenious norm delta %.3f",
                              i, loss, frob_delta)
    rm(Asvd, Bsvd)
    svd_old = svd_new
    if(frob_delta < convergence_tol) {
      futile.logger::flog.debug("reco:::soft_impute: converged with tol %f after %d iter", convergence_tol, i)
      CONVERGED = TRUE
      break
    }
  }
  setattr(svd_new, "trace", data.table::rbindlist(trace_iter))
  if(!CONVERGED)
    futile.logger::flog.warn("reco:::soft_impute: didn't converged with tol %f after %d iterations - returning latest solution",
                             convergence_tol, i)
  svd_new
}

# workhorse for soft_impute
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
  x_delta@x = x@x - make_sparse_approximation(x, A, B)
  loss = (as.numeric(crossprod(x_delta@x)) + lambda * sum(svd_current$d)) / length(x_delta@x)

  first = (x_delta %*% svd_current[[singular_vectors]]) %*% diag( sqrt(svd_current$d) / (svd_current$d + lambda))
  rm(x_delta)

  second = t(A * (svd_current$d / (svd_current$d + lambda)))
  res = reco:::svd_econ((first + second) %*% diag(sqrt(svd_current$d)))
  data.table::setattr(res, "loss", loss)
  res
}
