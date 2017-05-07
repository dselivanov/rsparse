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
                          n_cores =
                          init_stdv = 0.01) {
      private$set_internal_matrix_formats(sparse = "dgCMatrix", dense = NULL)
      private$lambda = lambda
      private$init_stdv = init_stdv
      private$rank = rank
      private$scorers = new.env(hash = TRUE, parent = emptyenv())
      if (n_cores > 1 && .Platform$OS.type != "unix") {
        flog.warn("Detected Windows platform and 'n_cores > 1'. This won't work
                  since library relies on fork-based parellelism. Setting n_cores = 1.")
        n_cores = 1
      }
      private$n_cores = n_cores
    },
    fit = function(x, n_iter = 5L, n_thread = private$n_cores, ...) {

      # x = 1 + alpha * r
      # x = confidense matrix, not ratings/interactions matrix!
      # we expect user already transformed it

      flog.debug("convert input to %s if needed", private$internal_matrix_formats$sparse)
      c_ui = private$check_convert_input(x, private$internal_matrix_formats)

      flog.debug("check items in input are not negative")
      stopifnot(all(c_ui@x >= 0))

      flog.debug("making antoher matrix for convenient traverse by users - transposing input matrix")
      c_iu = t(c_ui)

      # init
      nr = nrow(c_ui)
      nc = ncol(c_ui)

      private$U = matrix(rnorm(nr * private$rank, 0, private$init_stdv), ncol = nr, nrow = private$rank)
      private$I = matrix(rnorm(nc * private$rank, 0, private$init_stdv), ncol = nc, nrow = private$rank)
      Lambda = diag(x = private$lambda, nrow = private$rank, ncol = private$rank)

      trace_values = vector("numeric", n_iter)

      flog.info("starting factorization with %d workers", n_cores)
      trace_lst = vector("list", n_iter)
      # iterate
      for (i in seq_len(n_iter)) {

        private$IIt = tcrossprod(private$I) + Lambda
        flog.debug("iter %d by item", i)
        stopifnot(ncol(private$U) == ncol(c_iu))
        # private$U will be modified in place
        als_implicit(c_iu, private$I, private$IIt, private$U, nth = n_cores, ...)
        # private$U = private$solver(private$I, private$IIt, c_iu, n_cores = n_cores, ...)

        private$UUt = tcrossprod(private$U) + Lambda
        flog.debug("iter %d by user", i)
        stopifnot(ncol(private$I) == ncol(c_ui))

        # private$I will be modified in place
        als_implicit(c_ui, private$U, private$UUt, private$I, nth = n_cores, ...)
        # private$I = private$solver(private$U, private$UUt, c_ui, n_cores = n_cores, ...)

        # flog.debug("calculating loss")
        # trace_values[[i]] = calc_als_implicit_loss(c_ui, private$U, private$I, private$lambda, n_cores, ...)
        # gc()
        # trace_loss = sprintf("iter %d loss %.4f", i, trace_values[[i]])
        # trace_loss = sprintf("iter %d", i)

        #------------------------------------------------------------------------
        # calculate some metrics if needed in order to diagnose convergence
        #------------------------------------------------------------------------
        trace_iter = vector("list", length(names(private$scorers)))
        j = 1L
        trace_scors_string = ""
        for(sc in names(private$scorers)) {
          score = private$scorers[[sc]]()
          trace_scors_string = sprintf("%s score %s = %f", trace_scors_string, sc, score)
          trace_iter[[j]] = list(iter = i, scorer = sc, value = score)
          j = j + 1L
        }
        trace_lst[[i]] = rbindlist(trace_iter)
        flog.info("%s", trace_scors_string)
        #------------------------------------------------------------------------
      }

      private$components_ = private$I

      res = t(private$U)
      setattr(res, "trace", rbindlist(trace_lst))
      invisible(res)
    },
    fit_transform = function(x, n_iter = 5L, n_thread = private$n_cores, ...) {
      res = self$fit(x, n_iter, n_thread, ...)
      res
    },
    # project new user into latent user space
    factorize_new_users = function(x) {
      stopifnot(ncol(x) == ncol(private$I))
      as.matrix( t( solve(private$IIt, t(tcrossprod(x, private$I)) ) ) )
    },
    # project new items into latent item space
    factorize_new_items = function(x) {
      stopifnot(nrow(x) == ncol(private$U))
      as.matrix(solve(private$UUt, private$U %*% x))
    },
    predict = function(x, k, n_cores = private$n_cores, ...) {
      m = nrow(x)
      # transform user features into latent space
      x_latent_space = self$factorize_new_users(x)
      # calculate scores for each item
      x_similarity = x_latent_space %*% private$I
      res = parallel::mclapply(seq_len(m), function(i) {
        top_n(x_similarity[i, ], k)
      }, mc.cores = n_cores, ...)
      do.call(rbind, res)
    },
    # preds = predictions
    ap_k = function(x, y, k = ncol(x)) {
      stopifnot(ncol(x) == ncol(y))
      stopifnot(nrow(x) == nrow(y))
      n_u = nrow(x)
      preds = self$predict(x, k)
      y_csr = as(y, "RsparseMatrix")
      res = numeric(n_u)
      for(u in seq_len(n_u)) {
        p1 = y_csr@p[[u]]
        p2 = y_csr@p[[u + 1]]
        ind = p1 + seq_len(p2 - p1)
        u_ind = y_csr@j[ind] + 1L
        u_x = y_csr@x[ind]
        ord = order(u_x, decreasing = TRUE)
        res[[u]] = ap_at_k(preds[u, ], u_ind[ord], k = k)
      }
      res
    },
    ndcg_k = function(x, y, k = ncol(x)) {
      # stopifnot(ncol(x) == ncol(y))
      stopifnot(nrow(x) == nrow(y))
      n_u = nrow(x)
      preds = self$predict(x, k)
      y_csr = as(y, "RsparseMatrix")
      res = numeric(n_u)
      for(u in seq_len(n_u)) {
        p1 = y_csr@p[[u]]
        p2 = y_csr@p[[u + 1]]
        ind = p1 + seq_len(p2 - p1)
        u_ind = y_csr@j[ind] + 1L
        u_x = y_csr@x[ind]
        ord = order(u_x, decreasing = TRUE)
        res[[u]] = ndcg_at_k(preds[u, ], u_ind[ord], u_x[ord], k)
      }
      res
    },
    # x = sparse matrix of observed user interactions
    # y = sparse matrix of observed user we are trying to predtict
    add_scorer = function(x, y, name, metric, ...) {
      if(exists(name, where = private$scorers, inherits = FALSE))
        stop(sprintf("scorer with name '%s' already exists", name))

      if(metric == "loss") {
        private$scorers[[name]] = function() {
          # transpose since internally users kept as n_factor * n_users
          U_new = t(self$factorize_new_users(x))
          calc_als_implicit_loss(x, private$U, private$I, private$lambda, n_cores = private$n_cores, ...)
        }
      } else {
        if (length(grep(pattern = "(ndcg|map)\\@[[:digit:]]+", x = metric)) != 1 )
          stop(sprintf("don't know how add '%s' metric. Only 'loss', 'map@k', 'ndcg@k' are supported", metric))

        scorer_conf = strsplit(metric, "@", T)[[1]]
        k = as.integer(tail(scorer_conf, 1))
        scorer_fun = scorer_conf[[1]]

        if(scorer_fun == "map")
          private$scorers[[name]] = function() mean(self$ap_k(x, y, k))
        if(scorer_fun == "ndcg")
          private$scorers[[name]] = function() mean(self$ndcg_k(x, y, k))
      }
    },
    remove_scorer = function(name) {
      if(!exists(name, where = private$scorers))
        stop(sprintf("can't find scorer '%s'", name))
      rm(list = name, envir = private$scorers)
    },
    finalize = function() {
      rm(list = names(private$scorers), envir = private$scorers)
      private$scorers = NULL
      gc();
    }
  ),
  private = list(
    scorers = NULL,
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

    # R solver for reference. Now replaced with fast Armadillo solver
    solver = function(X, XtX_reg, C_UI, n_cores = private$n_cores, ...) {

      # XtX = tcrossprod(X)
      m_rank = nrow(X)

      # BLAS multithreading should be switched off
      # https://stat.ethz.ch/pipermail/r-sig-hpc/2012-July/001432.html
      # https://hyperspec.wordpress.com/2012/07/26/altering-openblas-threads/
      # https://stat.ethz.ch/pipermail/r-sig-debian/2016-August/002586.html

      nc = ncol(C_UI)

      # aliases to sparse matrix components for convenience
      C_UI_P = C_UI@p
      C_UI_I = C_UI@i
      C_UI_X = C_UI@x

      splits = parallel::splitIndices(nc, n_cores)

      RES = parallel::mclapply(splits, function(chunk) {
        # MAP PHASE
        chunk_len = length(chunk)
        chunk_start = chunk[[1]]
        chunk_end = chunk[[chunk_len]]
        n_print = as.integer(chunk_len / 16)

        RES_WORKER = matrix(data = 0, nrow = m_rank, ncol = chunk_len)

        # RES_WORKER = lapply(chunk, function(i) {

        for(j in seq_along(chunk)) {
          i = chunk[[j]]
          if(i %% n_print == 0) {
            flog.debug("worker %d progress %d%%", Sys.getpid(), as.integer(100 * (i - chunk_start) / chunk_len))
          }

          # column pointers
          p1 = C_UI_P[[i]]
          p2 = C_UI_P[[i + 1L]]
          pind = p1 + seq_len(p2 - p1)

          # add 1L because in C_UI indices are 0-based and in dense matrices they are 1-based
          ind = C_UI_I[pind] + 1L
          confidence = C_UI_X[pind]

          # corresponds to Y for a given user by we take only columns which have non-zero interactions
          # drop = FALSE - don't simplify matrices with 1 column to a vector
          X_nnz = X[, ind, drop = FALSE]

          # line below corresponds to [(YtY + lambda * I) + Yt %*% (C_u - I) %*% Y] in paper Hu, Koren, Volinsky paper
          # XtX_reg = (YtY + lambda * I)
          # confidence - 1 = C_u - I
          # X_nnz %*% (t.default(X_nnz) * (confidence - 1)) = Yt %*% (C_u - I) %*% Y

          inv = XtX_reg + X_nnz %*% (t.default(X_nnz) * (confidence - 1))
          rhs = X_nnz %*% confidence


          # same as RES_WORKER[, j] = solve(inv, rhs)
          # but doesn't perform many checks, doesn't copy attributes, etc. So it is notably faster.
          RES_WORKER[, j] = .Internal(La_solve(inv, rhs, .Machine$double.eps))

          #------------------------------------
          # what didn't work
          #------------------------------------
          # This was slower
          # X_nnz = t.default(X[, ind, drop = FALSE])
          # inv = XtX_reg + base::crossprod(X_nnz, X_nnz * (confidence - 1))
          # rhs = base::crossprod(X_nnz, confidence)

          # Cholesky was slower as well
          # RES_WORKER[, j] = chol2inv(chol.default(inv)) %*% rhs
          #------------------------------------
        }
        RES_WORKER
      }, mc.cores = n_cores, ...)
      # GLOBAL REDUCE
      do.call(cbind, RES)
    }
  )
)

# proxy loss since we don't calculate loss for "negative" (not observed) items
calc_als_implicit_loss = function(X, user, item, lambda = 0, n_cores = 1, ...) {
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
