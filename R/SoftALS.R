#' @title SoftImpute/SoftSVD matrix factorization
#' @description Fit SoftImpute/SoftSVD via fast alternating least squares. Based on the
#' paper by Trevor Hastie, Rahul Mazumder, Jason D. Lee, Reza Zadeh
#' by "Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares" -
#' \url{https://arxiv.org/pdf/1410.2596.pdf}
#' @param x sparse matrix. Both CSR \code{dgRMatrix} and CSC \code{dgCMatrix} are supported.
#' CSR matrix is preffered because in this case algorithm will benefit from multithreaded
#' CSR * dense matrix products (if OpenMP is supported on your platform).
#' On many-cores machines this reduces fitting time significantly.
#' @param rank maximum rank of the low-rank solution.
#' @param lambda regularization parameter for the nuclear norm
#' @param n_iter maximum number of iterations of the algorithms
#' @param convergence_tol convergence tolerance.
#' Internally functions keeps track of the relative change of the Frobenious norm
#' of the two consequent iterations. If the change is less than \code{convergence_tol}
#' then the process is considered as converged and function returns result.
#' @param init \link{svd} like object with \code{u, v, d} components to initialize algorithm.
#' Algorithm benefit from warm starts. \code{init} could be rank up \code{rank} of the maximum allowed rank.
#' If \code{init} has rank less than max rank it will be padded automatically.
#' @param final_svd \code{logical} whether need to make final preprocessing with SVD.
#' This is not necessary but cleans up rank nicely - hithly recommnded to leave it \code{TRUE}.
#' @return \link{svd}-like object - \code{list(u, v, d)}. \code{u, v, d}
#' components represent left, right singular vectors and singular values.
#' @export
#' @examples
#'\donttest{
#' set.seed(42)
#' data('movielens100k')
#' k = 10
#' seq_k = seq_len(k)
#' m = movielens100k[1:100, 1:200]
#' svd_ground_true = svd(m)
#' svd_soft_svd = soft_svd(m, rank = k, n_iter = 100, convergence_tol = 1e-6)
#' m_restored_svd = svd_ground_true$u[, seq_k]  %*%
#'    diag(x = svd_ground_true$d[seq_k]) %*%
#'    t(svd_ground_true$v[, seq_k])
#' m_restored_soft_svd = svd_soft_svd$u %*%
#'   diag(x = svd_soft_svd$d) %*%
#'   t(svd_soft_svd$v)
#' all.equal(m_restored_svd, m_restored_soft_svd, tolerance = 1e-1)
#'}
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
  x_delta@x = x@x - make_sparse_approximation(x, A, B)
  loss = (as.numeric(crossprod(x_delta@x)) + lambda * sum(svd_current$d)) / length(x_delta@x)

  logger$trace("[solve_iter_als_softimpute] calculating first part of result")
  first = (x_delta %*% svd_current[[singular_vectors]]) %*% diag( sqrt(svd_current$d) / (svd_current$d + lambda))
  rm(x_delta)

  logger$trace("[solve_iter_als_softimpute] calculating second part of result")
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
    logger$trace(sprintf("running iter %d of the %s", i, target))

    if(target == "soft_impute") {
      B_hat = solve_iter_als_softimpute(tx, svd_new, lambda, "u")
      B_hat = B_hat %*% diag(sqrt(svd_new$d))
    } else if(target == "svd") {
      B_hat = solve_iter_als_svd(tx, svd_new, lambda, "u")
    }
    Bsvd = svd(B_hat)
    rm(B_hat)
    svd_new$v = Bsvd$u
    svd_new$d = Bsvd$d
    # not sure why this line is required
    svd_new$u = svd_new$u %*% Bsvd$v
    rm(Bsvd)
    # 2. calculate for users
    if(target == "soft_impute") {
      A_hat = solve_iter_als_softimpute(x, svd_new, lambda, "v")
      A_hat = A_hat %*% diag(sqrt(svd_new$d))
    } else if(target == "svd") {
      A_hat = solve_iter_als_svd(x, svd_new, lambda, "v")
    }
    loss =  attr(A_hat, "loss")
    if(is.null(loss)) loss = NA_real_

    Asvd = svd(A_hat)
    rm(A_hat)

    svd_new$u = Asvd$u
    svd_new$d = Asvd$d
    # not sure why this line is required
    svd_new$v = svd_new$v %*% Asvd$v
    rm(Asvd)

    #log values of loss and change in frobenious norm
    frob_delta = calc_frobenius_norm_delta(svd_old, svd_new)

    trace_iter[[k]] = list(iter = i, scorer = "frob_delta", value = frob_delta)
    k = k + 1L
    trace_iter[[k]] = list(iter = i, scorer = "loss", value = loss)
    k = k + 1L

    logger$info(sprintf("soft_als: iter %03d, frobenious norm change %.3f loss %.3f ", i, frob_delta, loss))

    svd_old = svd_new
    # check convergence and
    if(frob_delta < convergence_tol) {
      logger$info("soft_impute: converged with tol %f after %d iter", convergence_tol, i)
      CONVERGED = TRUE
      break
    }
  }
  setattr(svd_new, "trace", data.table::rbindlist(trace_iter))
  if(!CONVERGED)
    logger$warn("soft_impute: hasn't converged with tol %f after %d iterations - returning latest solution",
                convergence_tol, i)

  if(final_svd) {
    logger$trace("running final svd")
    if(target == "soft_impute") {
      A = t(svd_new$u) * sqrt(svd_new$d)
      B = t(svd_new$v) * sqrt(svd_new$d)
      x@x = x@x - make_sparse_approximation(x, A, B)
      m = x %*% svd_new$v + t(A) %*% (B %*% svd_new$v)
    } else if(target == "svd") {
      m = x %*% svd_new$v
    }

    m_svd = svd(m)
    final_singular_values = pmax(m_svd$d - lambda, 0)
    # FIXME cast back to float because there is no pmax/pmin in float at the moment
    if(is_input_float) final_singular_values = fl(final_singular_values)

    n_nonzero_singular_values = sum(final_singular_values > 0)

    if(n_nonzero_singular_values == 0) {
      msg = sprintf("regularization lambda=%f is too high - all singular vectors are zero", lambda)
      logger$error(msg)
      stop(msg)
    } else {
      logger$trace("final rank = %d", n_nonzero_singular_values)
    }

    svd_new = list(d = final_singular_values[seq_len(n_nonzero_singular_values)],
                   u = m_svd$u[, seq_len(n_nonzero_singular_values)],
                   v = tcrossprod(svd_new$v, m_svd$v)[, seq_len(n_nonzero_singular_values)])
  }
  svd_new
}
