#' @name ALS_implicit
#'
#' @title ALS implicit
#' @description Creates ALS implicit model.
#' See (Hu, Koren, Volinsky)'2008 paper \url{http://yifanhu.net/PUB/cf.pdf} for details.
#' @seealso \url{https://math.stackexchange.com/questions/1072451/analytic-solution-for-matrix-factorization-using-alternating-least-squares/1073170#1073170}
#' \url{http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/}
#' \url{http://datamusing.info/blog/2015/01/07/implicit-feedback-and-collaborative-filtering/}
#' \url{https://jessesw.com/Rec-System/}
#' \url{http://danielnee.com/2016/09/collaborative-filtering-using-alternating-least-squares/}
#' \url{http://www.benfrederickson.com/matrix-factorization/}
#' \url{http://www.benfrederickson.com/fast-implicit-matrix-factorization/}
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

      # private$U = matrix(rnorm(nr * private$rank, 0, private$init_stdv), ncol = nr, private$rank)
      private$I = matrix(rnorm(nc * private$rank, 0, private$init_stdv), ncol = nc, private$rank)
      Lambda = diag(x = private$lambda, nrow = private$rank, ncol = private$rank)

      trace_values = vector("numeric", n_iter)

      flog.info("starting factorization with %d workers", n_cores)
      # iterate
      for (i in seq_len(n_iter)) {

        private$IIt = tcrossprod(private$I) + Lambda
        flog.info("iter %d by item", i)
        private$U = private$solver(private$I, private$IIt, c_ui, Lambda, n_cores = n_cores, ...)
        gc()

        private$UUt = tcrossprod(private$U) + Lambda
        flog.info("iter %d by user", i)
        private$I = private$solver(private$U, private$UUt, c_iu, Lambda, n_cores = n_cores, ...)
        gc()

        if(trace) {
          flog.info("calculating loss")
          trace_values[[i]] = calc_als_implicit_proxy_loss(c_iu, private$U, private$I, private$lambda, n_cores, ...)
          gc()
          flog.info("iter %d loss %.4f", i, trace_values[[i]])
          for(sc in names(private$scorers)) {
            flog.info("%s = %f", sc, private$scorers[[sc]]() )
          }
        }
      }

      private$components_ = private$I

      res = t(private$U)
      if(trace) setattr(res, "trace", trace_values)
      invisible(res)
    },
    fit_transform = function(x, n_iter = 5L, n_cores = 1, trace = FALSE, ...) {
      res = self$fit(x, n_iter, n_cores, trace, ...)
      res
    },
    # project new user into latent user space
    factorize_new_users = function(x) {
      stopifnot(ncol(x) == ncol(private$I))
      # XXt = tcrossprod(private$I)
      # as.matrix(t(solve(XXt, t(tcrossprod(x, private$I)))))
      # XXt = tcrossprod(private$I)
      as.matrix(t(solve(private$IIt, t(tcrossprod(x, private$I)))))
    },
    # project new items into latent item space
    factorize_new_items = function(x) {
      stopifnot(nrow(x) == ncol(private$U))
      # XXt = tcrossprod(private$U)
      # as.matrix(solve(XXt, private$U %*% x))
      # XXt = tcrossprod(private$U)
      as.matrix(solve(private$UUt, private$U %*% x))
    },
    predict = function(x, k) {
      m = nrow(x)
      # transform user features into latent space
      x_latent_space = self$factorize_new_users(x)
      # calculate scores for each item
      x_similarity = x_latent_space %*% private$I
      # allocate memory in advance
      res = matrix(NA_integer_, nrow = m, ncol = k)
      # sort predictions by score and fill into result matrix
      for(i in seq_len(m)) {
        top_k = order(x_similarity[i, ], decreasing = T)[seq_len(k)]
        res[i, ] = top_k
      }
      res
    },
    mapk = function(x, y, k) {
      stopifnot(ncol(x) == ncol(y))
      stopifnot(nrow(x) == nrow(y))
      n_u = nrow(x)
      preds = self$predict(x, k)
      y_csr = as(y, "RsparseMatrix")
      apk = numeric(n_u)
      for(u in seq_len(n_u)) {
        p1 = y_csr@p[[u]]
        p2 = y_csr@p[[u + 1]]
        ind = p1 + seq_len(p2 - p1)
        u_ind = y_csr@j[ind] + 1L
        u_x = y_csr@x[ind]
        apk[[u]] = average_precision_at_k(preds[u, ], u_ind[order(u_x)], k = k)
      }
      mean(apk)
    },
    # x = sparse matrix of observed user interactions
    # y = sparse matrix of observed user we are trying to predtict
    # set_scorer = function(x, y, metric = c("map@10", "ndcg@10", "rmse", "loss")) {
    add_scorer = function(x, y, name, metric = "map@10") {

      if(exists(name, envir = private$scorers))
        stop(sprintf("scorer with name '%s' already exists", name))

      k = as.integer(tail(strsplit(metric, "@", T)[[1]], 1))
      private$scorers[[name]] = function() self$mapk(x, y, k)
    },
    remove_scorer = function(name) {
      if(!exists(name, envir = private$scorers))
        stop(sprintf("can't find scorer '%s'", name))
      rm(name, envir = private$scorers)
    }
  ),
  private = list(
    scorers = new.env(hash = TRUE, parent = emptyenv()),
    lambda = NULL,
    init_stdv = NULL,
    rank = NULL,
    # user factor matrix = rank * n_users
    U = NULL,
    # users tcrossprod
    UUt = NULL,
    # item factor matrix = rank * n_items
    I = NULL,
    # items tcrossprod
    IIt = NULL,
    #------------------------------------------------------------
    # X = factor matrix n_factors * (n_users or n_items)
    # C_UI = user-item confidence matrix
    # Lambda = regularization diagonal matrix
    solver = function(X, XtX_reg, C_UI, Lambda, n_cores = 1, ...) {

      # XtX = tcrossprod(X)
      m_rank = nrow(X)
      # XtX_reg = XtX + Lambda

      nc = ncol(C_UI)
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
          p1 = C_UI@p[[i]]
          p2 = C_UI@p[[i + 1L]]
          pind = p1 + seq_len(p2 - p1)

          ind = C_UI@i[pind] + 1L
          confidence = C_UI@x[pind]

          C_UI_nnz = X[, ind, drop = FALSE]
          inv = XtX_reg + C_UI_nnz %*% (t(C_UI_nnz) * (confidence - 1))
          rhs = C_UI_nnz %*% confidence

          # This was slower
          # C_UI_nnz = t(X[, ind, drop = FALSE])
          # inv = XtX_reg + crossprod(C_UI_nnz, C_UI_nnz * (confidence - 1))
          # rhs = crossprod(C_UI_nnz, confidence)

          # Cholesky was slower as well
          # RES_WORKER[, j] = chol2inv(chol.default(inv)) %*% rhs

          # same as
          # RES_WORKER[, j] = solve(inv, rhs)
          # but make no check, no attributes copy, etc. So it notably faster
          RES_WORKER[, j] = .Internal(La_solve(inv, rhs, .Machine$double.eps))
        }
        RES_WORKER
      }, mc.cores = n_cores, ...)
      # GLOBAL REDUCE
      do.call(cbind, RES)
    }
  )
)

# proxy loss since we don't calculate loss for "negative" (not observed) items
calc_als_implicit_proxy_loss = function(X, user, item, lambda = 0, n_cores = 1, ...) {
  loss = parallel::mclapply(
    parallel::splitIndices(ncol(X), n_cores),
    function(ii) {
      loss_chunk = 0
      for(i in ii) {
        p1 = X@p[i]
        p2 = X@p[i + 1]
        p = p1 + seq_len(p2 - p1)
        ind = X@i[p] + 1L
        xx  = X@x[p]
        item_i = item[, i, drop = FALSE]
        user_i = user[, ind, drop = FALSE]
        loss_chunk = loss_chunk + sum( xx * ( ( 1 - crossprod(user_i, item_i) ) ^ 2 ) )
      }
      # flog.info("loss at worker %d = %.3f", Sys.getpid(), loss_chunk)
      loss_chunk
    },
    mc.cores = n_cores, ...)
  loss = sum(unlist(loss))
  # add regularization if needed
  if(lambda > 0) loss = loss + lambda * (sum(user ^ 2) + sum(item ^ 2))

  # loss per number of non zero interactions
  loss / length(X@x)
}
