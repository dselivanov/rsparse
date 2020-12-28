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
#'   \item{Zhou, Yunhong, et al.
#'         "Large-scale parallel collaborative filtering for the netflix prize."
#'         International conference on algorithmic applications in management.
#'         Springer, Berlin, Heidelberg, 2008.}
#'   \item{Liang, Dawen, et al.
#'         "Factorization meets the item embedding: Regularizing matrix factorization with item co-occurrence."
#'         Proceedings of the 10th ACM conference on recommender systems. 2016.}
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
    #' @param dynamic_lambda whether `lambda` is to be scaled according to the number
    #  of non-missing entries for each row/column (only applicable to the explicit-feedback model).
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
    #' @param with_user_item_bias \code{bool} controls if  model should calculate user and item biases.
    #' At the moment only implemented for \code{"explicit"} feedback.
    #' @param with_global_bias \code{bool} controls if model should calculate global biases (mean).
    #' At the moment only implemented for \code{"explicit"} feedback.
    #' @param with_implicit_features \code{bool} In the explicit feedback model, whether to jointly
    #' factorize a binary matrix of user/item occurrences.
    #' @param cg_steps \code{integer > 0} - max number of internal steps in conjugate gradient
    #' (if "conjugate_gradient" solver used). \code{cg_steps = 3} by default.
    #' Controls precision of linear equation solution at the each ALS step.
    #' Usually no need to tune this parameter.
    #' @param weight_implicit \code{numeric >= 0} When passing `with_implicit_features=TRUE`,
    #' this is the weight that the implicit features will have in the loss function.
    #' @param precision one of \code{c("double", "float")}. Should embedding matrices be
    #' numeric or float (from \code{float} package). The latter is usually 2x faster and
    #' consumes less RAM. BUT \code{float} matrices are not "base" objects. Use carefully.
    #' @param ... not used at the moment
    initialize = function(rank = 10L,
                          lambda = 0,
                          dynamic_lambda = TRUE,
                          init = NULL,
                          preprocess = identity,
                          feedback = c("implicit", "explicit"),
                          solver = c("conjugate_gradient", "cholesky", "nnls"),
                          with_user_item_bias = FALSE,
                          with_global_bias = FALSE,
                          with_implicit_features = FALSE,
                          cg_steps = 3L,
                          weight_implicit = 0.5,
                          precision = c("double", "float"),
                          ...) {
      stopifnot(is.null(init) || is.matrix(init))
      solver = match.arg(solver)
      private$non_negative = ifelse(solver == "nnls", TRUE, FALSE)
      feedback = match.arg(feedback)

      if (feedback == 'implicit') {
        # FIXME
        # now only support bias for explicit feedback
        with_user_item_bias = FALSE
        with_global_bias = FALSE
      }
      if (private$non_negative && with_global_bias == TRUE) {
        logger$warn("setting `with_global_bias=FALSE` for 'nnls' solver")
        with_global_bias = FALSE
      }
      private$with_user_item_bias = with_user_item_bias

      private$with_global_bias = with_global_bias
      self$global_bias = 0

      solver_codes = c("cholesky", "conjugate_gradient", "nnls")
      private$solver_code = match(solver, solver_codes) - 1L

      private$precision = match.arg(precision)
      private$feedback = feedback
      private$lambda = as.numeric(lambda)
      private$dynamic_lambda = as.logical(dynamic_lambda)[1L]
      private$with_implicit_features = as.logical(with_implicit_features)[1L]
      private$weight_implicit = as.numeric(weight_implicit)[1L]
      stopifnot(private$weight_implicit >= 0.)

      stopifnot(is.integer(cg_steps) && length(cg_steps) == 1)
      private$cg_steps = cg_steps

      n_threads = getOption("rsparse_omp_threads", 1L)
      private$solver = function(x, X, Y, is_bias_last_row, X_implicit = NULL, XtX = NULL,
                                cnt_X=NULL, avoid_cg = FALSE, XtX_implicit = NULL) {
        solver_use = ifelse(avoid_cg && private$solver_code == 1L, 0L, private$solver_code)
        if (private$lambda && dynamic_lambda && is.null(cnt_X)) {
          if (private$precision == "double") {
            cnt_X = numeric(ncol(X))
          } else {
            cnt_X = float::float(ncol(X))
          }
        } else if (is.null(cnt_X)) {
          cnt_X = numeric()
          if (private$precision == "float")
            cnt_X = float::fl(cnt_X)
        }
        if (is.null(XtX_implicit)) {
          XtX_implicit = matrix(numeric(), nrow=0L, ncol=0L)
          X_implicit = matrix(numeric(), nrow=0L, ncol=0L)
        }
        if (private$precision == "float" && !("float32" %in% class(XtX_implicit))) {
          XtX_implicit = float::fl(XtX_implicit)
          X_implicit = float::fl(X_implicit)
        }

        if(feedback == "implicit") {
          als_implicit(
            x, X, Y,
            lambda = private$lambda,
            n_threads = n_threads,
            solver_code = solver_use,
            cg_steps = private$cg_steps,
            precision = private$precision,
            with_user_item_bias = private$with_user_item_bias,
            is_bias_last_row = is_bias_last_row,
            XtX = XtX)
        } else {
          als_explicit(
            x, X, Y, X_implicit, XtX_implicit, cnt_X,
            lambda = private$lambda,
            n_threads = n_threads,
            solver_code = solver_use,
            cg_steps = private$cg_steps,
            dynamic_lambda = private$dynamic_lambda,
            with_implicit_features = private$with_implicit_features && nrow(XtX_implicit) && ncol(XtX_implicit),
            precision = private$precision,
            with_user_item_bias = private$with_user_item_bias,
            is_bias_last_row = is_bias_last_row)
        }

      }

      private$init_user_item_bias = function(c_ui, c_iu, user_bias, item_bias) {
        FUN = ifelse(private$precision == 'double',
                     initialize_biases_double,
                     initialize_biases_float)
        FUN(c_ui, c_iu, user_bias, item_bias, private$lambda, private$dynamic_lambda,
            private$non_negative, private$with_global_bias)
      }

      self$components = init
      if (private$with_user_item_bias) {
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

      # init
      n_user = nrow(c_ui)
      n_item = ncol(c_ui)

      logger$trace("initializing U")
      if (private$precision == "double") {
        private$U = large_rand_matrix(private$rank, n_user)
        # for item biases
        if (private$with_user_item_bias) {
          private$U[1, ] = rep(1.0, n_user)
        }

        if (private$with_implicit_features) {
          rank_implicit = private$rank - ifelse(private$with_user_item_bias, 0L, 2L)
          self$components_i = matrix(
            rnorm(n_item * rank_implicit, 0, 0.01),
            ncol = n_item,
            nrow = rank_implicit
          )
          U_i = matrix(
            rnorm(n_user * rank_implicit, 0, 0.01),
            ncol = n_user,
            nrow = rank_implicit
          )
        } else {
          self$components_i = NULL
          U_i = NULL
        }
      } else {
        private$U = flrnorm(private$rank, n_user, 0, 0.01)
        if (private$with_user_item_bias) {
          private$U[1, ] = float::fl(rep(1.0, n_user))
        }

        if (private$with_implicit_features) {
          rank_implicit = private$rank - ifelse(private$with_user_item_bias, 0L, 2L)
          self$components_i = flrnorm(rank_implicit, n_item, 0, 0.01)
          U_i = flrnorm(rank_implicit, n_user, 0, 0.01)
        } else {
          self$components_i = NULL
          U_i = NULL
        }
      }

      if (is.null(self$components)) {
        if (private$precision == "double") {
          self$components = large_rand_matrix(private$rank, n_item)
          # for user biases
          if (private$with_user_item_bias) {
            self$components[private$rank, ] = rep(1.0, n_item)
          }
        } else {
          self$components = flrnorm(private$rank, n_item, 0, 0.01)
          if (private$with_user_item_bias) {
            self$components[private$rank, ] = float::fl(rep(1.0, n_item))
          }
        }
      } else {
        stopifnot(is.matrix(self$components) || is.float(self$components))
        stopifnot(ncol(self$components) == n_item)
        stopifnot(nrow(self$components) == private$rank)
      }

      if (!private$with_implicit_features)
        XtX_implicit_u = NULL

      # NNLS
      if (private$non_negative) {
        self$components = abs(self$components)
        private$U = abs(private$U)
      }

      stopifnot(ncol(private$U) == ncol(c_iu))
      stopifnot(ncol(self$components) == ncol(c_ui))

      if (private$with_user_item_bias) {
        logger$debug("initializing biases")
        # copy only c_ui@x
        # because c_iu is internal
        if (private$feedback == "explicit" && private$with_global_bias)
          c_ui@x = deep_copy(c_ui@x)


        if (private$precision == "double") {
          user_bias = numeric(n_user)
          item_bias = numeric(n_item)
        } else {
          user_bias = float(n_user)
          item_bias = float(n_item)
        }

        self$global_bias = private$init_user_item_bias(c_ui, c_iu, user_bias, item_bias)
        self$components[1L, ] = item_bias
        private$U[private$rank, ] = user_bias
      } else if (private$feedback == "explicit" && private$with_global_bias) {
        self$global_bias = mean(c_ui@x)
        c_ui@x = c_ui@x - self$global_bias
        c_iu@x = c_iu@x - self$global_bias
      }

      logger$info("starting factorization with %d threads", getOption("rsparse_omp_threads", 1L))

      loss_prev_iter = Inf

      # for dynamic lambda, need to keep track of the number of entries
      # in order to calculate the regularized loss
      cnt_u = numeric()
      cnt_i = numeric()
      if (private$dynamic_lambda) {
        cnt_u = as.numeric(diff(c_ui@p))
        cnt_i = as.numeric(diff(c_iu@p))
      }
      if (private$precision == "float") {
        cnt_u = float::fl(cnt_u)
        cnt_i = float::fl(cnt_i)
      }

      # iterate
      for (i in seq_len(n_iter)) {
        if (private$with_implicit_features) {
          self$components_i = solver_implicit_features(c_ui, private$U, private$lambda,
                                                       private$dynamic_lambda, private$with_user_item_bias,
                                                       private$non_negative)
          U_i = solver_implicit_features(c_iu, self$components, private$lambda,
                                         private$dynamic_lambda, private$with_user_item_bias,
                                         private$non_negative)
          private$XtX_implicit = private$weight_implicit * tcrossprod(self$components_i)
          XtX_implicit_u = private$weight_implicit * tcrossprod(U_i)
          if (private$weight_implicit != 1.) {
            self$components_i = private$weight_implicit * self$components_i
            U_i = private$weight_implicit * U_i
          }
        }

        # solve for items
        loss = private$solver(c_ui, private$U, self$components, TRUE, cnt_X=cnt_i,
                              X_implicit=U_i, XtX_implicit = XtX_implicit_u)
        # solve for users
        loss = private$solver(c_iu, self$components, private$U, FALSE, cnt_X=cnt_u,
                              X_implicit=self$components_i, XtX_implicit = private$XtX_implicit)

        logger$info("iter %d loss = %.4f", i, loss)
        if (loss_prev_iter / loss - 1 < convergence_tol) {
          logger$info("Converged after %d iterations", i)
          break
        }

        loss_prev_iter = loss
      }

      rank_ = ifelse(private$with_user_item_bias, private$rank - 1L, private$rank)
      ridge = fl(diag(x = private$lambda, nrow = rank_, ncol = rank_))

      X = if (private$with_user_item_bias) tcrossprod(self$components[-1L, ]) else self$components
      private$XtX = tcrossprod(X) + ridge
      if (private$precision == "float" && private$with_implicit_features) {
        self$components_i = float::fl(self$components_i)
        private$XtX_implicit = float::fl(private$XtX_implicit)
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
      if (self$global_bias != 0.)
        x@x = x@x - self$global_bias

      if (private$precision == "double") {
        res = matrix(0, nrow = private$rank, ncol = nrow(x))
      } else {
        res = float(0, nrow = private$rank, ncol = nrow(x))
      }

      loss = private$solver(t(x), self$components, res, FALSE, self$components_i, private$XtX,
                            avoid_cg=TRUE, XtX_implicit=private$XtX_implicit)

      res = t(res)

      if (private$precision == "double")
        setattr(res, "dimnames", list(rownames(x), NULL))
      else
        setattr(res@Data, "dimnames", list(rownames(x), NULL))

      res
    },
    #' @field components_i Components from implicit features (if passing
    #' `with_implicit_features=TRUE`).
    components_i = NULL
  ),
  #### private -----
  private = list(
    solver_code = NULL,
    cg_steps = NULL,
    scorers = NULL,
    lambda = NULL,
    dynamic_lambda = FALSE,
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
    XtX_implicit = NULL,
    solver = NULL,
    with_user_item_bias = NULL,
    with_global_bias = NULL,
    with_implicit_features = FALSE,
    weight_implicit = 0.,
    init_user_item_bias = NULL
  )
)

als_implicit = function(
  x, X, Y,
  lambda,
  n_threads,
  solver_code,
  cg_steps,
  precision,
  with_user_item_bias,
  is_bias_last_row,
  XtX = NULL) {

  solver = ifelse(precision == "float",
                  als_implicit_float,
                  als_implicit_double)

  if(is.null(XtX)) {
    rank = ifelse(with_user_item_bias, nrow(X) - 1L, nrow(X))
    ridge = fl(diag(x = lambda, nrow = rank, ncol = rank))
    if (with_user_item_bias) {
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
  x, X, Y, X_implicit, XtX_implicit, cnt_X,
  lambda,
  n_threads,
  solver_code,
  cg_steps,
  dynamic_lambda,
  with_implicit_features,
  precision,
  with_user_item_bias,
  is_bias_last_row) {

  solver = ifelse(precision == "float",
                  als_explicit_float,
                  als_explicit_double)

  # Y is modified in-place
  loss = solver(x, X, Y, X_implicit, XtX_implicit, cnt_X,
                lambda, n_threads, solver_code, cg_steps,
                dynamic_lambda, with_implicit_features,
                with_user_item_bias, is_bias_last_row)
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
  # loss = als_loss_explicit(x, X, res, lambda, getOption("rsparse_omp_threads", 1L))
  # data.table::setattr(res, "loss", loss)
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
  # loss = als_loss_explicit(x, X, res, lambda, getOption("rsparse_omp_threads", 1L))
  # data.table::setattr(res, "loss", loss)
  res
}

solver_implicit_features = function(x, X, lambda = 0, dynamic_lambda = TRUE, with_user_item_bias = FALSE, non_negative = FALSE) {
  if (with_user_item_bias)
    X = X[seq(2L, nrow(X) - 1L), , drop=FALSE]
  ridge = diag(x = lambda * ifelse(dynamic_lambda, ncol(X), 1), nrow = nrow(X), ncol = nrow(X))
  x@x = rep(1., length(x@x))
  if (!non_negative) {
    return(solve(tcrossprod(X) + ridge, X %*% x))
  } else {
    return(c_nnls_double(tcrossprod(X) + ridge, X %*% x, 10000L, 1e-3))
  }
}
