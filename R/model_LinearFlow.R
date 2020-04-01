#' @title Linear-FLow model for one-class collaborative filtering
#' @description Creates \emph{Linear-FLow} model described in
#' \href{http://www.bkveton.com/docs/ijcai2016.pdf}{Practical Linear Models for Large-Scale One-Class Collaborative Filtering}.
#' The goal is to find item-item (or user-user) similarity matrix which is \bold{low-rank and has small Frobenius norm}. Such
#' double regularization allows to better control the generalization error of the model.
#' Idea of the method is somewhat similar to \bold{Sparse Linear Methods(SLIM)} but scales to large datasets much better.
#' @references
#' \itemize{
#'   \item{\url{http://www.bkveton.com/docs/ijcai2016.pdf}}
#'   \item{\url{http://www-users.cs.umn.edu/~xning/slides/ICDM2011_slides.pdf}}
#' }
#' @export
#' @examples
#' data("movielens100k")
#' train = movielens100k[1:900, ]
#' cv = movielens100k[901:nrow(movielens100k), ]
#' model = LinearFlow$new(
#'   rank = 10, lambda = 0,
#'   solve_right_singular_vectors = "svd"
#' )
#' user_emb = model$fit_transform(train)
#' preds = model$predict(cv, k = 10)
LinearFlow = R6::R6Class(
  classname = "LinearFlow",
  inherit = MatrixFactorizationRecommender,
  public = list(
    #' @field v right singular vector of the user-item matrix. Size is \code{n_items * rank}.
    #' In the paper this matrix is called \bold{v}
    v = NULL,
    #' @description creates Linear-FLow model with \code{rank} latent factors.
    #' @param rank size of the latent dimension
    #' @param lambda regularization parameter
    #' @param init initialization of the orthogonal basis.
    #' @param preprocess \code{identity()} by default. User spectified function which will
    #' be applied to user-item interaction matrix before running matrix factorization
    #' (also applied during inference time before making predictions).
    #' For example we may want to normalize each row of user-item matrix to have 1 norm.
    #' Or apply \code{log1p()} to discount large counts.
    #' @param solve_right_singular_vectors type of the solver for initialization of the orthogonal
    #' basis. Original paper uses SVD. See paper for details.
    initialize = function(rank = 8L,
                          lambda = 0,
                          init = NULL,
                          preprocess = identity,
                          solve_right_singular_vectors = c("soft_impute", "svd")) {
      private$preprocess = preprocess
      private$rank = as.integer(rank)
      private$solve_right_singular_vectors = match.arg(solve_right_singular_vectors)
      private$lambda = as.numeric(lambda)
      self$v = init
    },
    #' @description performs matrix factorization
    #' @param x input matrix
    #' @param ... not used at the moment
    fit_transform = function(x, ...) {
      stopifnot(inherits(x, "sparseMatrix") || inherits(x, "SparsePlusLowRank"))
      x = private$preprocess(x)
      private$item_ids = colnames(x)
      self$v = private$get_right_singular_vectors(x, ...)
      logger$trace("calculating RHS")

      # rhs = t(self$v) %*% t(x) %*% x
      # same as above but a bit faster:
      rhs = crossprod(x %*% self$v, x)

      logger$trace("calculating LHS")
      lhs = rhs %*% self$v
      self$components = private$fit_transform_internal(lhs, rhs, private$lambda, ...)
      invisible(as.matrix(x %*% self$v))
    },
    #' @description calculates user embeddings for the new input
    #' @param x input matrix
    #' @param ... not used at the moment
    transform = function(x, ...) {
      stopifnot(inherits(x, "sparseMatrix") || inherits(x, "SparsePlusLowRank"))
      x = private$preprocess(x)
      res = x %*% self$v
      if (!is.matrix(res))
        res = as.matrix(res)
      invisible(res)
    },
    #' @description performs fast tuning of the parameter `lambda` with warm re-starts
    #' @param x input user-item interactions matrix. Model performs matrix facrtorization based
    #' only on this matrix
    #' @param x_train user-item interactions matrix. Model recommends items based on this matrix.
    #' Usually should be different from `x` to avoid overfitting
    #' @param x_test target user-item interactions. Model will evaluate predictions against this
    #' matrix, `x_test` should be treated as future interactions.
    #' @param lambda numeric vector - sequaence of regularization parameters. Supports special
    #' value like `auto@10`. This will automatically fine a sequence of lambda of length 10. This
    #' is recommended way to check for `lambda`.
    #' @param metric a metric against which model will be evaluated for top-k recommendations.
    #' Currently only \code{map@@k} and \code{ndcg@@k} are supported (\code{k} can be any integer)
    #' @param not_recommend matrix same shape as `x_train`. Specifies which items to not recommend
    #' for each user.
    #' @param ... not used at the moment
    cross_validate_lambda = function(x, x_train, x_test, lambda = "auto@10", metric = "map@10",
                  not_recommend = x_train, ...) {

      private$item_ids = colnames(x)
      stopifnot(inherits(not_recommend, "sparseMatrix") || is.null(not_recommend))
      if (inherits(not_recommend, "sparseMatrix"))
        not_recommend = as(not_recommend, "RsparseMatrix")

      stopifnot(private$item_ids == colnames(x_test))
      stopifnot(private$item_ids == colnames(x_train))

      x = private$preprocess(x)
      x_train = private$preprocess(x_train)

      lambda_auto = FALSE
      if (is.character(lambda)) {
        if (length(grep(pattern = "(auto)\\@[[:digit:]]+", x = lambda)) != 1 )
          stop(sprintf("don't know how add '%s' metric 'auto@k' or numeric are supported", lambda))
        lambda = strsplit(lambda, "@", T)[[1]]
        lambdas_k = as.integer(lambda[[2]])
        lambda_auto = TRUE
      } else {
        stopifnot(is.numeric(lambda))
      }

      if (length(grep(pattern = "(ndcg|map)\\@[[:digit:]]+", x = metric)) != 1 )
        stop(sprintf("don't know how add '%s' metric. Only 'map@k', 'ndcg@k' are supported", metric))
      metric = strsplit(metric, "@", T)[[1]]
      metric_k = as.integer(metric[[2]])
      metric_name = metric[[1]]

      self$v = private$get_right_singular_vectors(x, ...)
      logger$trace("calculating RHS")
      # rhs = t(self$v) %*% t(x) %*% x
      # same as above but a bit faster:
      rhs = crossprod(x %*% self$v, x)

      logger$trace("calculating LHS")
      lhs = rhs %*% self$v
      # calculate "reasonable" lambda from values of main diagonal of LHS
      if (lambda_auto) {
        lhs_ridge = diag(lhs)
        # generate sequence of lambda
        lambda = seq(log10(0.1 * min(lhs_ridge)), log10(10 * max(lhs_ridge)), length.out = lambdas_k)
        lambda = 10 ^ lambda
      }

      cv_res = data.frame(lambda = lambda, score = NA_real_)
      xq_cv_train = as.matrix(x_train %*% self$v)

      for (i in seq_along(lambda)) {
        lambda_i = lambda[[i]]
        Y = private$fit_transform_internal(lhs, rhs, lambda_i, ...)
        # preds = private$predict_internal(xq_cv_train, k = metric_k, Y = Y, not_recommend = not_recommend)
        preds = private$predict_internal(xq_cv_train, Y, k = metric_k, not_recommend = not_recommend)
        score = NULL
        if (metric_name == "map")
          score = mean(ap_k(preds, x_test, ...), na.rm = T)
        if (metric_name == "ndcg")
          score = mean(ndcg_k(preds, x_test, ...), na.rm = T)

        cv_res$score[[i]] = score
        if (score >= max(cv_res$score, na.rm = T) || is.null(self$components)) {
          self$components = Y
          private$lambda = lambda_i
        }
        logger$trace("%d/%d lambda %.3f score = %.3f", i, length(lambda), lambda_i, score)
      }
      cv_res
    }
  ),
  private = list(
    rank = NULL,
    preprocess = NULL,
    solve_right_singular_vectors = NULL,
    lambda = NULL,
    # item_ids = NULL,
    get_right_singular_vectors = function(x, ...) {
      result = NULL
      if (!is.null(self$v)) {
        logger$trace("found `init`, checking it")
        stopifnot(nrow((self$v)) == ncol(x))
        stopifnot(ncol((self$v)) == private$rank)
        result = self$v
      } else {
        if (is.null(self$v)) {
          if (private$solve_right_singular_vectors == "soft_impute")
            trunc_svd = soft_impute(x, rank = private$rank, lambda = 0, ...)
          else if (private$solve_right_singular_vectors == "svd")
            trunc_svd = soft_svd(x, rank = private$rank, lambda = 0, ...)
          else
            stop(sprintf("don't know solver '%s'", private$solve_right_singular_vectors))
        }
        result = trunc_svd$v
      }
      stopifnot(is.numeric(result))
      result
    },
    fit_transform_internal = function(lhs, rhs, lambda, ...) {
      logger$trace("solving least squares with lambda %.3f", lambda)
      lhs_ridge = lhs + diag(rep(lambda, private$rank))
      as.matrix(solve(lhs_ridge, rhs))
    }
  )
)
