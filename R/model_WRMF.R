#' @title Weighted Regularized Matrix Factorization for collaborative filtering
#' @description Creates a matrix factorization model which is solved through Alternating Least Squares (Weighted ALS for implicit feedback).
#' For implicit feedback see "Collaborative Filtering for Implicit Feedback Datasets" (Hu, Koren, Volinsky).
#' For explicit feedback it corresponds to the classic model for rating matrix decomposition with MSE error.
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
#'   \item{Franc, Vojtech, Vaclav Hlavac, and Mirko Navara.
#'         "Sequential coordinate-wise algorithm for the
#'         non-negative least squares problem."
#'         International Conference on Computer Analysis of Images
#'         and Patterns. Springer, Berlin, Heidelberg, 2005.}
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
    #' @param solver \code{character} - solver name.
    #' One of \code{c("conjugate_gradient", "cholesky", "nnls")}.
    #' Usually approximate \code{"conjugate_gradient"} is significantly faster and solution is
    #' on par with \code{"cholesky"}.
    #' \code{"nnls"} performs non-negative matrix factorization (NNMF) - restricts
    #' user and item embeddings to be non-negative.
    #' @param with_bias \code{bool} controls model should calculate user and item biases.
    #' At the moment only implemented for \code{"explicit"} feedback.
    #' @param cg_steps \code{integer > 0} - max number of internal steps in conjugate gradient
    #' (if "conjugate_gradient" solver used). \code{cg_steps = 3} by default.
    #' Controls precision of linear equation solution at the each ALS step. Usually no need to tune this parameter
    #' @param precision one of \code{c("double", "float")}. Should embedding matrices be
    #' numeric or float (from \code{float} package). The latter is usually 2x faster and
    #' consumes less RAM. BUT \code{float} matrices are not "base" objects. Use carefully.
    #' @param ... not used at the moment
    initialize = function(rank = 10L,
                          lambda = 0,
                          init = NULL,
                          preprocess = identity,
                          feedback = c("implicit", "explicit"),
                          solver = c("conjugate_gradient", "cholesky", "nnls"),
                          with_bias = FALSE,
                          cg_steps = 3L,
                          precision = c("double", "float"),
                          ...) {
      stopifnot(is.null(init) || is.matrix(init))
      solver = match.arg(solver)
      private$non_negative = ifelse(solver == "nnls", TRUE, FALSE)

      precision = match.arg(precision)
      feedback = match.arg(feedback)

      if (feedback == 'implicit') {
        # FIXME
        # now only support bias for explicit feedback
        with_bias = FALSE
      }
      private$with_bias = with_bias

      solver_codes = c("cholesky", "conjugate_gradient", "nnls")
      private$solver_code = match(solver, solver_codes) - 1L

      private$precision = match.arg(precision)
      private$feedback = feedback
      private$lambda = as.numeric(lambda)

      stopifnot(is.integer(cg_steps) && length(cg_steps) == 1)
      private$cg_steps = cg_steps

      n_threads = getOption("rsparse_omp_threads", 1L)
      private$solver = function(x, X, Y, is_bias_last_row, XtX = NULL) {
        if(feedback == "implicit") {
          als_implicit(
            x, X, Y,
            lambda = private$lambda,
            n_threads = n_threads,
            solver_code = private$solver_code,
            cg_steps = private$cg_steps,
            precision = private$precision,
            with_bias = private$with_bias,
            is_bias_last_row = is_bias_last_row,
            XtX = XtX)
        } else {
          als_explicit(
            x, X, Y,
            lambda = private$lambda,
            n_threads = n_threads,
            solver_code = private$solver_code,
            cg_steps = private$cg_steps,
            precision = private$precision,
            with_bias = private$with_bias,
            is_bias_last_row = is_bias_last_row)
        }

      }

      if (solver == "conjugate_gradient" && feedback == "explicit")
        logger$warn("only 'cholesky' is available for 'explicit' feedback")

      self$components = init
      if (private$with_bias) {
        private$rank = as.integer(rank) + 2L
      } else {
        private$rank = as.integer(rank)
      }


      stopifnot(is.function(preprocess))
      private$preprocess = preprocess
    },
    #' @description fits the model
    #' @param x input matrix (preferably matrix  in CSC format -`CsparseMatrix`
    #' @param n_iter max number of ALS iterations
    #' @param convergence_tol convergence tolerance checked between iterations
    #' @param ... not used at the moment
    fit_transform = function(x, n_iter = 10L, convergence_tol = ifelse(private$feedback == "implicit", 0.005, 0.001), ...) {
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
      c_iu = t(c_ui)
      # store item_ids in order to use them in predict method
      private$item_ids = colnames(c_ui)

      if ((private$feedback != "explicit") || private$non_negative) {
        stopifnot(all(c_ui@x >= 0))
      }

      # if (private$feedback == "explicit" && !private$non_negative) {
      #   self$global_mean = mean(c_ui@x)
      #   c_ui@x = c_ui@x - self$global_mean
      # }
      # if (private$with_bias) {
      #   c_ui@x = deep_copy(c_ui@x)
      #   c_ui_orig = deep_copy(c_ui@x)
      # }
      # else {
      #   c_ui_orig = numeric(0L)
      # }

      # if (private$with_bias) {
      #   c_iu_orig = deep_copy(c_iu@x)
      # } else {
      #   c_iu_orig = numeric(0L)
      # }

      # init
      n_user = nrow(c_ui)
      n_item = ncol(c_ui)

      logger$trace("initializing U")
      if (private$precision == "double") {
        private$U = matrix(
          runif(n_user * private$rank, 0, 0.01),
          ncol = n_user,
          nrow = private$rank
        )
        # for item biases
        if (private$with_bias) {
          private$U[1, ] = rep(1.0, n_user)
        }
      } else {
        private$U = flrunif(private$rank, n_user, 0, 0.01)
        if (private$with_bias) {
          private$U[1, ] = float::fl(rep(1.0, n_user))
        }
      }

      if (is.null(self$components)) {
        if (private$precision == "double") {
          self$components = matrix(
            runif(n_item * private$rank, 0, 0.01),
            ncol = n_item,
            nrow = private$rank
          )
          # for user biases
          if (private$with_bias) {
            self$components[private$rank, ] = rep(1.0, n_item)
          }
        } else {
          self$components = flrunif(private$rank, n_item, 0, 0.01)
          if (private$with_bias) {
            self$components[private$rank, ] = float::fl(rep(1.0, n_item))
          }
        }
      } else {
        stopifnot(is.matrix(self$components) || is.float(self$components))
        stopifnot(ncol(self$components) == n_item)
        stopifnot(nrow(self$components) == private$rank)
      }

      # NNLS
      if (private$non_negative) {
        self$components = abs(self$components)
        private$U = abs(private$U)
      }

      stopifnot(ncol(private$U) == ncol(c_iu))
      stopifnot(ncol(self$components) == ncol(c_ui))

      # if (private$with_bias) {
      #   logger$debug("initializing biases")
      #   if (private$precision == "double") {
      #     user_bias = numeric(n_user)
      #     item_bias = numeric(n_item)
      #     initialize_biases_double(c_ui, c_iu,
      #                              user_bias,
      #                              item_bias,
      #                              private$lambda,
      #                              private$non_negative)
      #   } else {
      #     user_bias = float(n_user)
      #     item_bias = float(n_item)
      #     initialize_biases_float(c_ui, c_iu,
      #                             user_bias,
      #                             item_bias,
      #                             private$lambda,
      #                             private$non_negative)
      #   }
      #   self$components[1L, ] = item_bias
      #   private$U[private$rank, ] = user_bias
      # }

      logger$info("starting factorization with %d threads", getOption("rsparse_omp_threads", 1L))

      loss_prev_iter = Inf

      # iterate
      for (i in seq_len(n_iter)) {
        # solve for items
        loss = private$solver(c_ui, private$U, self$components, TRUE)
        # solve for users
        loss = private$solver(c_iu, self$components, private$U, FALSE)

        logger$info("iter %d loss = %.4f", i, loss)
        if (loss_prev_iter / loss - 1 < convergence_tol) {
          logger$info("Converged after %d iterations", i)
          break
        }

        loss_prev_iter = loss
      }

      rank_ = ifelse(private$with_bias, private$rank - 1L, private$rank)
      ridge = fl(diag(x = private$lambda, nrow = rank_, ncol = rank_))

      X = if (private$with_bias) tcrossprod(self$components[-1L, ]) else self$components
      private$XtX = tcrossprod(X) + ridge

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

      loss = private$solver(t(x), self$components, res, FALSE, private$XtX)
      # if (private$feedback == "implicit") {
      #   loss = private$solver(t(x), self$components, res, FALSE, private$XtX)
      # } else{
      #   x_use = t(x)
      #   if (!private$non_negative)
      #     x_use@x = x_use@x - self$global_mean
      #   if (private$with_bias) {
      #     x_orig = deep_copy(x_use@x)
      #   } else {
      #     x_orig = numeric(0L)
      #   }
      #   loss = private$solver(x_use, self$components, res, FALSE)
      # }
      res = t(res)

      if (private$precision == "double")
        setattr(res, "dimnames", list(rownames(x), NULL))
      else
        setattr(res@Data, "dimnames", list(rownames(x), NULL))

      res
    }
  ),
  #### private -----
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
    precision = NULL,
    XtX = NULL,
    solver = NULL,
    with_bias = NULL
  )
)

als_implicit = function(
  x, X, Y,
  lambda,
  n_threads,
  solver_code,
  cg_steps,
  precision,
  with_bias,
  is_bias_last_row,
  XtX = NULL) {

  solver = ifelse(precision == "float",
                  als_implicit_float,
                  als_implicit_double)

  if(is.null(XtX)) {
    rank = ifelse(with_bias, nrow(X) - 1L, nrow(X))
    ridge = fl(diag(x = lambda, nrow = rank, ncol = rank))
    if (with_bias) {
      index_row_to_discard = ifelse(is_bias_last_row, rank, 1L)
      XtX = tcrossprod(X[-index_row_to_discard, ])
    } else {
      XtX = tcrossprod(X)
    }
    XtX = XtX + ridge
  }
  # Y is modified in-place
  loss = solver(x, X, Y, XtX, lambda, n_threads, solver_code, cg_steps, is_bias_last_row)
}

als_explicit = function(
  x, X, Y, XtX,
  lambda,
  n_threads,
  solver_code,
  cg_steps,
  precision,
  with_bias,
  is_bias_last_row) {

  solver = ifelse(precision == "float",
                  als_explicit_float,
                  als_explicit_double)

  # Y is modified in-place
  loss = solver(x, X, Y, lambda, n_threads, solver_code, cg_steps, with_bias, is_bias_last_row)
}

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

solver_explicit_biases = function(x, X, Y, bias_index = 1L, lambda = 0, non_negative = FALSE) {
  ones = rep(1.0, ncol(Y))
  y_bias_index = bias_index
  x_bias_index = setdiff(c(1, 2), y_bias_index)

  biases = X[x_bias_index, ]
  res = vector("list", ncol(x))
  ridge = diag(x = lambda, nrow = nrow(X) - 1L, ncol = nrow(X) - 1L)
  for (i in seq_len(ncol(x))) {
    # find non-zero ratings
    p1 = x@p[[i]]
    p2 = x@p[[i + 1L]]
    j = p1 + seq_len(p2 - p1)
    # and corresponding indices
    ind_nnz = x@i[j] + 1L
    x_nnz = x@x[j] - biases[ind_nnz]
    X_nnz =  X[-x_bias_index, ind_nnz, drop = F]
    XtX = tcrossprod(X_nnz) + ridge
    if (non_negative) {
      res[[i]] = c_nnls_double(XtX, X_nnz %*% x_nnz, 10000L, 1e-3)
    } else {
      res[[i]] = solve(XtX, X_nnz %*% x_nnz)
    }
  }
  res = do.call(cbind, res)
  if (y_bias_index == 1) {
    res = rbind(res[1, ], ones, res[-1, ], deparse.level = 0 )
  } else {
    res = rbind(ones, res, deparse.level = 0 )
  }
  loss = als_loss_explicit(x, X, res, lambda, getOption("rsparse_omp_threads", 1L))
  data.table::setattr(res, "loss", loss)
  res
}
