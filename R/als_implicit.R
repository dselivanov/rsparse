#' @name ALS_implicit
#'
#' @title ALS implicit
#' @description Creates ALS implicit model.
#' See (Hu, Koren, Volinsky)'2008 paper \url{http://yifanhu.net/PUB/cf.pdf} for details.
#' @format \code{\link{R6Class}} object.
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#' asl_impl = ALS_implicit$new(rank = 10L, lambda = 0, init_stdv = ifelse(lambda == 0, 0.01, 1 / sqrt(2 * lambda)))
#' asl_impl$fit(x, n_iter = 5L, n_cores = 1, trace = FALSE, ...)
#' asl_impl$fit_transform(x, n_iter = 5L, n_cores = 1, trace = FALSE, ...)
#' asl_impl$components
#' }
#' @section Methods:
#' \describe{
#'   \item{\code{$new(rank = 10L, lambda = 0, init_stdv = ifelse(lambda == 0, 0.01, 1 / sqrt(2 * lambda)))}}{
#'     create ALS_implicit model with \code{rank} latent factors}
#'   \item{\code{$fit(x, n_iter = 5L, n_cores = 1, trace = FALSE, ...)}}{
#'     fit model to an input user-item \bold{confidence!} matrix. (preferably in "dgCMatrix" format)}.
#'     \code{x} should be a confidence matrix which corresponds to \code{1 + alpha * r_ui} in original paper.
#'     Usually \code{r_ui} cosrresponds to the number of interactions of user \code{u} and item \code{i}.
#'   \item{\code{$fit_transform(x, n_iter = 5L, n_cores = 1, trace = FALSE, ...)}}{Explicitly returns factor matrix for
#'     users of size \code{n_users * rank}. See description for \code{fit} above.}.
#'   \item{\code{$components}}{item factors matrix of size \code{rank * n_items}}.
#'}
#' @section Arguments:
#' \describe{
#'  \item{asl_impl}{A \code{ALS_implicit} model.}
#'  \item{x}{An input user-item \bold{confidence} matrix.}
#'  \item{rank}{\code{integer} desired number of latent factors}
#'  \item{lambda}{\code{numeric} regularization parameter}
#'  \item{init_stdv}{\code{numeric} std dev for initialization of the factor matrices}
#'  \item{trace}{\code{logical} whether to calculate loss proxy. By "proxy" we mean that we will calculate
#'  loss only for user-item pairs with observer interactions}
#'  \item{n_cores}{\code{n_cores} number of cores to use in factorization. Corresponds to
#'    \code{mc.cores} in \code{parallel::mclapply}. Ignored for Windows OS (with warning).}
#'  \item{...}{Arguments useful for \code{fit(), fit_transform()} -
#'  these arguments will be passed to \link{parallel::mclapply} function which is used for parallelization
#'  of computations.}
#' }
#' @export
ALS_implicit = R6::R6Class(
  inherit = mlapi::mlDecomposition,
  classname = "AlternatingLeastSquaresImplicit",
  public = list(
    initialize = function(rank = 10L,
                          lambda = 0,
                          init_stdv = ifelse(lambda == 0, 0.01, 1 / sqrt(2 * lambda))) {
      private$set_internal_matrix_formats(sparse = "dgCMatrix", dense = NULL)
      private$lambda = lambda
      private$init_stdv = init_stdv
      private$rank = rank
    },
    fit = function(x, n_iter = 5L, n_cores = 1, trace = FALSE, ...) {

      if (n_cores > 1 && .Platform$OS.type != "unix") {
        flog.warn("Detected Windows platform and 'n_cores > 1'. This won't work
                  since library relies on fork-based parellelism. Setting n_cores = 1.")
      }

      # x = 1 + alpha * r
      # x = confidense matrix, not ratings/interactions matrix!
      # we expect user already transformed it

      flog.debug("convert input to %s if needed", private$internal_matrix_formats$sparse)
      c_iu = private$check_convert_input(x, private$internal_matrix_formats)

      flog.debug("check items in input are not negative")
      stopifnot(all(c_iu@x >= 0))

      flog.debug("making antoher matrix for convenient traverse by users - transposing input matrix")
      c_ui = t(c_iu)

      # init
      nr = nrow(c_iu)
      nc = ncol(c_iu)

      X = matrix(rnorm(nr * private$rank, 0, private$init_stdv), ncol = nr, private$rank)
      Y = matrix(rnorm(nc * private$rank, 0, private$init_stdv), ncol = nc, private$rank)
      Lambda = diag(x = private$lambda, nrow = private$rank, ncol = private$rank)

      trace_values = vector("numeric", n_iter)

      flog.info("starting factorization with %d workers", n_cores)
      # iterate
      for (i in seq_len(n_iter)) {
        flog.info("iter %d by item", i)
        X = private$solver(Y, c_ui, Lambda, n_cores = n_cores, ...)
        gc()
        flog.info("iter %d by user", i)
        Y = private$solver(X, c_iu, Lambda, n_cores = n_cores, ...)
        gc()
        if(trace) {
          flog.debug("started calculation of loss proxy")
          trace_values[[i]] = calc_als_implicit_proxy_loss(x, X, Y)
          gc()
          flog.debug("iter %d proxy_loss %.4f", i, trace_values[[i]])
        }
      }

      private$components_ = Y
      res = t(X)
      if(trace) {
        setattr(res, "trace", trace_values)
      }
      invisible(res)
    },
    fit_transform = function(x, n_iter = 5L, n_cores = 1, trace = FALSE, ...) {
      res = self$fit(x, n_iter, n_cores, trace, ...)
      res
    }
  ),
  private = list(
    lambda = NULL,
    init_stdv = NULL,
    rank = NULL,
    #------------------------------------------------------------
    # M = factor matrix
    # Z = user-item confidence matrix
    # Lambda = regularization "eye" matrix
    solver = function(M, Z, Lambda, n_cores = 1, ...) {

      MtM = tcrossprod(M)
      m_rank = nrow(M)

      MtM_reg = MtM + Lambda
      nc = ncol(Z)
      # BLAS multithreading should be switched off
      # https://stat.ethz.ch/pipermail/r-sig-hpc/2012-July/001432.html
      # https://hyperspec.wordpress.com/2012/07/26/altering-openblas-threads/
      # https://stat.ethz.ch/pipermail/r-sig-debian/2016-August/002586.html
      splits = parallel::splitIndices(nc, n_cores)
      RES = parallel::mclapply(splits, function(chunk) {
        # MAP PHASE
        chunk_len = length(chunk)
        chunk_start = chunk[[1]]
        chunk_end = chunk[[chunk_len]]

        RES_WORKER = matrix(data = 0, nrow = m_rank, ncol = chunk_len)

        # RES_WORKER = lapply(chunk, function(i) {

        for(j in seq_along(chunk)) {
          i = chunk[[j]]
          if(i %% as.integer(chunk_len / 16) == 0) {
            flog.debug("worker %d progress %d%%", Sys.getpid(), as.integer(100 * (i - chunk_start) / chunk_len))
          }

          # column pointers
          p1 = Z@p[[i]]
          p2 = Z@p[[i + 1L]]
          pind = p1 + seq_len(p2 - p1)

          ind = Z@i[pind] + 1L
          confidence = Z@x[pind]

          Z_nnz = M[, ind, drop = FALSE]

          inv = MtM_reg + Z_nnz %*% (t(Z_nnz) * (confidence - 1))
          rhs = Z_nnz %*% confidence

          # inv = MtM_reg + Z_nnz %*% (t(Z_nnz) * confidence)
          # rhs = Z_nnz %*% (confidence + 1)
          # solve(inv, rhs)
          RES_WORKER[, j] = solve(inv, rhs)
        }
        #)
        # LOCAL REDUCE/AGGREGATE
        # RES_WORKER = do.call(cbind, RES_WORKER)
        RES_WORKER
      }, mc.cores = n_cores, ...)
      # GLOBAL REDUCE
      do.call(cbind, RES)
    }
  )
)

# proxy loss since we don't calculate loss for "negative" items
calc_als_implicit_proxy_loss = function(X, user, item, lambda = 0) {
  loss = 0
  for(i in 1:ncol(X)) {
    # if(i %% 25000 == 0) flog.info("%d", i)
    p1 = X@p[i]
    p2 = X@p[i+1]
    p = p1 + seq_len(p2 - p1)
    ind = X@i[p] + 1L
    xx = X@x[p]
    item_i = item[, i, drop = FALSE]
    user_i = user[, ind, drop = FALSE]
    loss = loss + sum( xx * ( ( 1 - crossprod(user_i, item_i) ) ^ 2 ) )
    loss
  }
  # add regularization
  loss = loss + lambda * (sum(user ^ 2) + sum(item ^ 2))
  # loss per number of non zero interactions
  loss / length(X@x)
}

# calc_als_implicit_proxy_loss = compiler::cmpfun(calc_als_implicit_proxy_loss)
