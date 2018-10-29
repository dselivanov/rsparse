#' @title SoftImpute/SoftSVD matrix factorization
#' @description Fit SoftImpute/SoftSVD via fast alternating least squares. Based on the
#' paper by Trevor Hastie, Rahul Mazumder, Jason D. Lee, Reza Zadeh
#' by "Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares" -
#' \url{https://arxiv.org/pdf/1410.2596.pdf}
#' @param x sparse matrix. Both CSR \code{dgRMatrix} and CSC \code{dgCMatrix} are supported.
#' in case of CSR matrix we suggest to load \url{https://github.com/dselivanov/MatrixCSR} package
#' which provides multithreaded CSR*dense matrix products (if OpenMP is supported on your platform).
#' On many-cores machines this reduces fitting time significantly.
#' @param rank maximum rank of the low-rank solution.
#' @param lambda regularization parameter for nuclear norm
#' @param n_iter maximum number of iterations of the algorithms
#' @param convergence_tol convergence tolerance.
#' Internally we keep track relative change of frobenious norm of two consequent iterations.
#' @param init \link{svd} like object with \code{u, v, d} components to initialize algorithm.
#' Algorithm benefit from warm starts. \code{init} could be rank up \code{rank} of the maximum allowed rank.
#' If \code{init} has rank less than max rank it will be padded automatically.
#' @param final_svd \code{logical} whether need to make final preprocessing with SVD.
#' This is not necessary but cleans up rank nicely - hithly recommnded to leave it \code{TRUE}.
#' @return \link{svd}-like object - \code{list} with \code{u, v, d}
#' components - left, right singular vectors and singular vectors.
#' @export
soft_impute = function(x,
                       rank = 10L, lambda = 0,
                       n_iter = 100L, convergence_tol = 1e-3,
                       init = NULL, final_svd = TRUE) {
  soft_als(x,
          rank = rank, lambda = lambda,
          n_iter = n_iter, convergence_tol = convergence_tol,
          init = init, final_svd = final_svd,
          target = "soft_impute")
}


#' @rdname soft_impute
#' @export
soft_svd = function(x,
                    rank = 10L, lambda = 0,
                    n_iter = 100L, convergence_tol = 1e-3,
                    init = NULL, final_svd = TRUE) {
  soft_als(x,
           rank = rank, lambda = lambda,
           n_iter = n_iter, convergence_tol = convergence_tol,
           init = init, final_svd = final_svd,
           target = "svd")
}

#--------------------------------------------------------------------------------------------
# workhorse for soft_impute
#--------------------------------------------------------------------------------------------
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
  futile.logger::flog.debug("soft-impute: 'make_sparse_approximation'")
  x_delta@x = x@x - make_sparse_approximation(x, A, B)
  futile.logger::flog.debug("soft-impute: calculating loss")
  loss = (as.numeric(crossprod(x_delta@x)) + lambda * sum(svd_current$d)) / length(x_delta@x)

  futile.logger::flog.debug("soft-impute: calculating first part of result")
  first = (x_delta %*% svd_current[[singular_vectors]]) %*% diag( sqrt(svd_current$d) / (svd_current$d + lambda))
  rm(x_delta)

  futile.logger::flog.debug("soft-impute: calculating second part of result")
  second = t(A * (svd_current$d / (svd_current$d + lambda)))
  res = first + second
  data.table::setattr(res, "loss", loss)
  res
}

#--------------------------------------------------------------------------------------------
# workhorse for soft_svd
#--------------------------------------------------------------------------------------------
solve_iter_als_svd = function(x, svd_current, lambda, singular_vectors = c("u", "v")) {
  singular_vectors = match.arg(singular_vectors)
  (x %*% svd_current[[singular_vectors]]) %*% diag((svd_current$d / (svd_current$d + lambda)))
}

#--------------------------------------------------------------------------------------------
# core EM-like algorithm for soft-svd and soft-impute
#--------------------------------------------------------------------------------------------
soft_als = function(x,
                    rank = 10L, lambda = 0,
                    n_iter = 100L, convergence_tol = 1e-3,
                    init = NULL, final_svd = TRUE,
                    target = c("svd", "soft_impute")) {

  target = match.arg(target)
  stopifnot(is.logical(final_svd) && length(final_svd) == 1)
  is_input_float = inherits(x, "float32")
  if(is_input_float) lambda = float::fl(lambda)
  tx = t(x)
  if(is.null(init)) {
    # draw random matrix and make columns orthogonal with QR decomposition
    U = matrix(rnorm(n = nrow(x) * rank), nrow = nrow(x))
    if(is_input_float) U = fl(U)

    U = qr.Q(qr(U, LAPACK = TRUE))

    # FIXME - to be addressed after fix in upstream https://github.com/wrathematics/float/issues/27
    if(is_input_float) U = fl(U)

    # init with dummy values
    D  = rep(1, rank)
    if(is_input_float) D = fl(D)

    V = matrix(rep(0, ncol(x) * rank), nrow = ncol(x))
    if(is_input_float) V = fl(V)

    svd_old = list(d = D, u = U, v = V); rm(U, V)
  } else {
    # warm start with another SVD
    stopifnot(is.list(init))
    stopifnot(all(names(init) %in% c("u", "d", "v")))
    if(length(init$d) > rank)
      stop("provided initial svd 'init' has bigger rank than model rank")
    svd_old = pad_svd(init, rank)
  }

  trace_iter = vector("list", n_iter)
  k = 1L

  svd_new = svd_old
  CONVERGED = FALSE
  for(i in seq_len(n_iter)) {
    # Alternating algorithm
    # 1. calculate for items
    if(target == "soft_impute") {
      futile.logger::flog.debug("running 'solve_iter_als_softimpute'")
      B_hat = solve_iter_als_softimpute(tx, svd_new, lambda, "u")
      futile.logger::flog.debug("running 'svd'")
      Bsvd = svd(B_hat %*% diag(sqrt(svd_new$d)))
    } else if(target == "svd") {
      futile.logger::flog.debug("running 'solve_iter_als_svd'")
      B_hat = solve_iter_als_svd(tx, svd_new, lambda, "u")
      futile.logger::flog.debug("running 'svd'")
      Bsvd = svd(B_hat)
    }
    rm(B_hat)
    svd_new$v = Bsvd$u
    svd_new$d = Bsvd$d
    # not sure why this line is required
    svd_new$u = svd_new$u %*% Bsvd$v

    # 2. calculate for users
    if(target == "soft_impute") {
      futile.logger::flog.debug("running 'solve_iter_als_softimpute'")
      A_hat = solve_iter_als_softimpute(x, svd_new, lambda, "v")
      futile.logger::flog.debug("running 'svd'")
      Asvd = svd(A_hat %*% diag(sqrt(svd_new$d)))
    } else if(target == "svd") {
      futile.logger::flog.debug("running 'solve_iter_als_svd'")
      A_hat = solve_iter_als_svd(x, svd_new, lambda, "v")
      futile.logger::flog.debug("running 'svd'")
      Asvd = svd(A_hat)
    } else {
      stop(sprintf("unknown target = %s", target))
    }
    loss =  attr(A_hat, "loss")
    rm(A_hat)

    svd_new$u = Asvd$u
    svd_new$d = Asvd$d
    # not sure why this line is required
    svd_new$v = svd_new$v %*% Asvd$v
    rm(Asvd, Bsvd)
    #log values of loss and change in frobenious norm
    futile.logger::flog.debug("running 'calc_frobenius_norm_delta'")
    frob_delta = calc_frobenius_norm_delta(svd_old, svd_new)
    trace_iter[[k]] = list(iter = i, scorer = "frob_delta", value = frob_delta)
    k = k + 1L
    if(!is.null(loss)) {
      trace_iter[[k]] = list(iter = i, scorer = "loss", value = loss)
      k = k + 1L
      futile.logger::flog.info("soft_als: iter %03d, loss %.3f frobenious norm change %.3f",
                                i, loss, frob_delta)
    } else {
      futile.logger::flog.info("soft_als: iter %03d, frobenious norm change %.3f",
                                i, frob_delta)
    }

    svd_old = svd_new
    # check convergence and
    if(frob_delta < convergence_tol) {
      futile.logger::flog.info("soft_impute: converged with tol %f after %d iter", convergence_tol, i)
      CONVERGED = TRUE
      break
    }
  }
  setattr(svd_new, "trace", data.table::rbindlist(trace_iter))
  if(!CONVERGED)
    futile.logger::flog.warn("soft_impute: didn't converged with tol %f after %d iterations - returning latest solution",
                             convergence_tol, i)
  if(final_svd) {
    futile.logger::flog.info("running final svd")
    if(target == "soft_impute") {
      A = t(svd_new$u) * sqrt(svd_new$d)
      B = t(svd_new$v) * sqrt(svd_new$d)
      x@x = x@x - make_sparse_approximation(x, A, B)
      m = x %*% svd_new$v + t(A) %*% (B %*% svd_new$v)
    } else if(target == "svd") {
      m = x %*% svd_new$v
    } else {
      stop(sprintf("unknown target = %s", target))
    }


    m_svd = svd(m)
    final_singular_values = pmax(m_svd$d - lambda, 0)
    # FIXME cast back to float because there is no pmax/pmin in float at the moment
    if(is_input_float) final_singular_values = fl(final_singular_values)

    n_nonzero_singular_values = sum(final_singular_values > 0)

    if(n_nonzero_singular_values == 0) {
      stop(sprintf("regularization lambda=%f is too high - all singular vectors are zero", lambda))
    } else {
      futile.logger::flog.info("final rank = %d", n_nonzero_singular_values)
    }

    svd_new = list(d = final_singular_values[seq_len(n_nonzero_singular_values)],
                   u = m_svd$u[, seq_len(n_nonzero_singular_values)],
                   v = tcrossprod(svd_new$v, m_svd$v)[, seq_len(n_nonzero_singular_values)])
  }
  svd_new
}
