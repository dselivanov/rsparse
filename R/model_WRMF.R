#' @title Weighted Regularized Matrix Factorization for collaborative filtering
#' @description Creates a matrix factorization model which is solved through Alternating Least Squares (Weighted ALS for implicit feedback).
#' For implicit feedback see "Collaborative Filtering for Implicit Feedback Datasets" (Hu, Koren, Volinsky).
#' For explicit feedback it corresponds to the classic model for rating matrix decomposition with MSE error (without biases at the moment).
#' These two algorithms are proven to work well in recommender systems.
#' @references
#' \itemize{
#'   \item{Hu, Yifan, Yehuda Koren, and Chris Volinsky.
#'         "Collaborative filtering for implicit feedback datasets."
#'         2008 Eighth IEEE International Conference on Data Mining. Ieee, 2008.}
#'   \item{\url{https://math.stackexchange.com/questions/1072451/analytic-solution-for-matrix-factorization-using-alternating-least-squares/1073170#1073170}}
#'   \item{\url{http://activisiongamescience.github.io/2016/01/11/Implicit-Recommender-Systems-Biased-Matrix-Factorization/}}
#'   \item{\url{http://datamusing.info/blog/2015/01/07/implicit-feedback-and-collaborative-filtering/}}
#'   \item{\url{https://jessesw.com/Rec-System/}}
#'   \item{\url{http://danielnee.com/2016/09/collaborative-filtering-using-alternating-least-squares/}}
#'   \item{\url{http://www.benfrederickson.com/matrix-factorization/}}
#'   \item{\url{http://www.benfrederickson.com/fast-implicit-matrix-factorization/}}
#' }
#' @export
#' @examples
#' data('movielens100k')
#' train = movielens100k[1:900, ]
#' cv = movielens100k[901:nrow(movielens100k), ]
#' model = WRMF$new(rank = 5,  lambda = 0, feedback = 'implicit')
#' user_emb = model$fit_transform(train, n_iter = 5, convergence_tol = -1)
#' item_emb = model$components
#' preds = model$predict(cv, k = 10, not_recommend = cv)
WRMF = R6::R6Class(
  inherit = MatrixFactorizationRecommender,
  classname = "WRMF",

  public = list(
    #' @description creates WRMF model
    #' @param rank size of the latent dimension
    #' @param lambda regularization parameter
    #' @param init initialization of item embeddings
    #' @param preprocess \code{identity()} by default. User spectified function which will
    #' be applied to user-item interaction matrix before running matrix factorization
    #' (also applied during inference time before making predictions).
    #' For example we may want to normalize each row of user-item matrix to have 1 norm.
    #' Or apply \code{log1p()} to discount large counts.
    #' This corresponds to the "confidence" function from
    #' "Collaborative Filtering for Implicit Feedback Datasets" paper.
    #' Note that it will not automatically add +1 to the weights of the positive entries.
    #' @param feedback \code{character} - feedback type - one of \code{c("implicit", "explicit")}
    #' @param non_negative logical, whether to perform non-negative factorization
    #' @param solver \code{character} - solver for "implicit feedback" problem.
    #' One of \code{c("conjugate_gradient", "cholesky")}.
    #' Usually approximate \code{"conjugate_gradient"} is significantly faster and solution is
    #' on par with \code{"cholesky"}
    #' @param cg_steps \code{integer > 0} - max number of internal steps in conjugate gradient
    #' (if "conjugate_gradient" solver used). \code{cg_steps = 3} by default.
    #' Controls precision of linear equation solution at the each ALS step. Usually no need to tune this parameter
    #' @param precision one of \code{c("double", "float")}. Should embeeding matrices be
    #' numeric or float (from \code{float} package). The latter is usually 2x faster and
    #' consumes less RAM. BUT \code{float} matrices are not "base" objects. Use carefully.
    #' @param ... not used at the moment
    initialize = function(rank = 10L,
                          lambda = 0,
                          init = NULL,
                          preprocess = identity,
                          feedback = c("implicit", "explicit"),
                          non_negative = FALSE,
                          solver = c("conjugate_gradient", "cholesky"),
                          cg_steps = 3L,
                          precision = c("double", "float"),
                          ...) {
      stopifnot(is.null(init) || is.matrix(init))
      solver = match.arg(solver)
      precision = match.arg(precision)
      feedback = match.arg(feedback)

      if (solver == "cholesky") private$solver_code = 0L
      if (solver == "conjugate_gradient") private$solver_code = 1L

      if (feedback == "explicit" && precision == "float")
        stop("Explicit solver doesn't support single precision at the moment (but in principle can support).")

      private$precision = match.arg(precision)
      private$feedback = feedback
      private$lambda = as.numeric(lambda)

      stopifnot(is.integer(cg_steps) && length(cg_steps) == 1)
      private$cg_steps = cg_steps

      private$non_negative = non_negative

      n_threads = getOption("rsparse_omp_threads", 1L)
      private$solver = function(x, X, Y, XtX) {
        if (feedback == "implicit") {
          solver_implicit(
            x, X, Y, XtX,
            lambda = private$lambda,
            n_threads = n_threads,
            solver_code = private$solver_code,
            cg_steps = private$cg_steps,
            non_negative = private$non_negative,
            precision = private$precision)
        } else {
          solver_explicit(x, X, Y, private$lambda, private$non_negative)
        }
      }

      if (solver == "conjugate_gradient" && feedback == "explicit")
        logger$warn("only 'cholesky' is available for 'explicit' feedback")

      self$components = init
      private$rank = as.integer(rank)

      stopifnot(is.function(preprocess))
      private$preprocess = preprocess
    },
    #' @description fits the model
    #' @param x input matrix (preferably matrix  in CSC format -`CsparseMatrix`
    #' @param n_iter max number of ALS iterations
    #' @param convergence_tol convergence tolerance checked between iterations
    #' @param ... not used at the moment
    fit_transform = function(x, n_iter = 10L, convergence_tol = 0.005, ...) {
      if (private$feedback == "implicit" ) {
        logger$trace("WRMF$fit_transform(): calling `RhpcBLASctl::blas_set_num_threads(1)` (to avoid thread contention)")
        blas_threads_keep = RhpcBLASctl::blas_get_num_procs()
        RhpcBLASctl::blas_set_num_threads(1)
        on.exit({
          logger$trace("WRMF$fit_transform(): on exit `RhpcBLASctl::blas_set_num_threads(%d)", blas_threads_keep)
          RhpcBLASctl::blas_set_num_threads(blas_threads_keep)
        })
      }

      c_ui = as(x, "CsparseMatrix")
      c_ui = private$preprocess(c_ui)
      # store item_ids in order to use them in predict method
      private$item_ids = colnames(c_ui)

      if ((private$feedback != "explicit") || private$non_negative) {
        stopifnot(all(c_ui@x >= 0))
      }
      c_iu = t(c_ui)

      # init
      n_user = nrow(c_ui)
      n_item = ncol(c_ui)

      logger$trace("initializing U")
      if (private$precision == "double")
        private$U = matrix(
          rnorm(n_user * private$rank, 0, 0.01),
          ncol = n_user,
          nrow = private$rank
        )
      else
        private$U = flrnorm(private$rank, n_user, 0, 0.01)

      if (is.null(self$components)) {
        if (private$precision == "double")
          self$components = matrix(
            rnorm(n_item * private$rank, 0, 0.01),
            ncol = n_item,
            nrow = private$rank
          )
        else
          self$components = flrnorm(private$rank, n_item, 0, 0.01)
      } else {
        stopifnot(is.matrix(self$components) || is.float(self$components))
        stopifnot(ncol(self$components) == n_item)
        stopifnot(nrow(self$components) == private$rank)
      }

      stopifnot(ncol(private$U) == ncol(c_iu))
      stopifnot(ncol(self$components) == ncol(c_ui))

      logger$info("starting factorization")
      loss_prev_iter = Inf

      # iterate
      for (i in seq_len(n_iter)) {
        # solve for items
        YtY = tcrossprod(private$U) +
          fl(diag(x = private$lambda, nrow = private$rank, ncol = private$rank))
        self$components = private$solver(c_ui, private$U, self$components, YtY)
        loss = attr(self$components, "loss")

        # solve for users
        private$XtX = tcrossprod(self$components) +
          fl(diag(x = private$lambda, nrow = private$rank, ncol = private$rank))
        private$U = private$solver(c_iu, self$components, private$U, private$XtX)
        loss = attr(private$U, "loss")

        logger$info("iter %d loss = %.4f", i, loss)
        if (loss_prev_iter / loss - 1 < convergence_tol) {
          logger$info("Converged after %d iterations", i)
          break
        }

        loss_prev_iter = loss
      }

      if (private$precision == "double")
        data.table::setattr(self$components, "dimnames", list(NULL, colnames(x)))
      else
        data.table::setattr(self$components@Data, "dimnames", list(NULL, colnames(x)))

      res = t(private$U)
      private$U = NULL

      if (private$precision == "double")
        setattr(res, "dimnames", list(rownames(x), NULL))
      else
        setattr(res@Data, "dimnames", list(rownames(x), NULL))
      res
    },
    # project new users into latent user space - just make ALS step given fixed items matrix
    #' @description create user embeddings for new input
    #' @param x user-item iteraction matrix
    #' @param ... not used at the moment
    transform = function(x, ...) {
      stopifnot(ncol(x) == ncol(self$components))
      if (private$feedback == "implicit" ) {
        logger$trace("WRMF$transform(): calling `RhpcBLASctl::blas_set_num_threads(1)` (to avoid thread contention)")
        blas_threads_keep = RhpcBLASctl::blas_get_num_procs()
        RhpcBLASctl::blas_set_num_threads(1)
        on.exit({
          logger$trace("WRMF$transform(): on exit `RhpcBLASctl::blas_set_num_threads(%d)", blas_threads_keep)
          RhpcBLASctl::blas_set_num_threads(blas_threads_keep)
        })
      }

      x = as(x, "CsparseMatrix")
      x = private$preprocess(x)

      if (private$precision == "double") {
        res = matrix(0, nrow = private$rank, ncol = nrow(x))
      } else {
        res = float(0, nrow = private$rank, ncol = nrow(x))
      }

      loss = private$solver(t(x), self$components, res, private$XtX)
      res = t(res)

      if (private$precision == "double")
        setattr(res, "dimnames", list(rownames(x), NULL))
      else
        setattr(res@Data, "dimnames", list(rownames(x), NULL))

      res
    }
  ),
  private = list(
    solver_code = NULL,
    cg_steps = NULL,
    scorers = NULL,
    lambda = NULL,
    rank = NULL,
    non_negative = NULL,
    # user factor matrix = rank * n_users
    U = NULL,
    # item factor matrix = rank * n_items
    I = NULL,
    # preprocess - transformation of input matrix before passing it to ALS
    # for example we can scale each row or apply log() to values
    # this is essentially "confidence" transformation from WRMF article
    preprocess = NULL,
    feedback = NULL,
    cv_data = NULL,
    scorers_ellipsis = NULL,
    precision = NULL,
    XtX = NULL,
    als_implicit_fun = NULL,
    solver = NULL
  )
)

solver_explicit = function(x, X, Y, lambda = 0, non_negative = FALSE) {
  res = vector("list", ncol(x))
  ridge = diag(x = lambda, nrow = nrow(X), ncol = nrow(X))
  for (i in seq_len(ncol(x))) {
    # find non-zero ratings
    p1 = x@p[[i]]
    p2 = x@p[[i + 1L]]
    j = p1 + seq_len(p2 - p1)
    x_nnz = x@x[j]
    # and corresponding indices
    ind_nnz = x@i[j] + 1L

    X_nnz = X[, ind_nnz, drop = F]
    XtX = tcrossprod(X_nnz) + ridge
    if (non_negative) {
      res[[i]] = c_nnls_double(XtX, X_nnz %*% x_nnz, 10000L, 1e-3)
    } else {
      res[[i]] = solve(XtX, X_nnz %*% x_nnz)
    }
  }
  res = do.call(cbind, res)
  loss = als_loss_explicit(x, X, res, lambda, getOption("rsparse_omp_threads", 1L))
  data.table::setattr(res, "loss", loss)
  res
}

solver_implicit = function(
  x, X, Y, XtX,
  lambda,
  n_threads,
  solver_code,
  cg_steps,
  non_negative,
  precision) {

  solver = ifelse(precision == "float",
         als_implicit_float,
         als_implicit_double)

  # Y is modified in-place
  loss = solver(x, X, Y, XtX, lambda, n_threads, solver_code, cg_steps, non_negative)
  res = Y
  data.table::setattr(res, "loss", loss)
  res
}
