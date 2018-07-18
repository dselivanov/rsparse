#' @name PureSVD
#'
#' @title Soft-SVD decompomposition
#' @description Creates matrix factorization model based on Soft-SVD.
#' Soft SVD is very similar to truncated SVD with ability do add regularization
#' based on nuclear norm.
#' @format \code{R6Class} object.
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#'   model = PureSVD$new(rank = 10L,
#'                       lambda = 0,
#'                       init = NULL,
#'                       preprocess = identity,
#'                       ...)
#'   model$fit_transform(x, n_iter = 5L, ...)
#'   model$predict(x, k, not_recommend = x, ...)
#'   model$components
#' }
#' @section Methods:
#' \describe{
#'   \item{\code{$new(rank = 10L, lambda = 0,
#'                    init = NULL,
#'                    preprocess = identity,
#'                    ...
#'                    ) }}{ creates matrix
#'     factorization model model with at most \code{rank} latent factors. If \code{init} is not null then initializes
#'     with provided SVD solution}
#'   \item{\code{$fit_transform(x, n_iter = 5L, ...)}}{ fits model to
#'     an input user-item matrix.
#'     \bold{Returns factor matrix for users of size \code{n_users * rank}}}
#'   \item{\code{$predict(x, k, not_recommend = x, ...)}}{predict \code{top k}
#'     item ids for users \code{x} (= column names from the matrix passed to \code{fit_transform()} method).
#'     Users features should be defined the same way as they were defined in training data - as \bold{sparse matrix}
#'     of confidence values (implicit feedback) or ratings (explicit feedback).
#'     Column names (=item ids) should be in the same order as in the \code{fit_transform()}.}
#'   \item{\code{$components}}{item factors matrix of size \code{rank * n_items}}
#'}
#' @section Arguments:
#' \describe{
#'  \item{model}{A \code{PureSVD} model.}
#'  \item{x}{An input sparse user-item matrix(of class \code{dgCMatrix})}.
#'  \item{rank}{\code{integer} - maximum number of latent factors}
#'  \item{lambda}{\code{numeric} - regularization parameter for nuclear norm}
#'  \item{preprocess}{\code{function} = \code{identity()} by default. User spectified function which will be applied to user-item interaction matrix
#'     before running matrix factorization (also applied in inference time before making predictions). For example we may
#'     want to normalize each row of user-item matrix to have 1 norm. Or apply \code{log1p()} to discount large counts.}
#'  \item{not_recommend}{\code{sparse matrix} or \code{NULL} - points which items should be excluided from recommendations for a user.
#'    By default it excludes previously seen/consumed items.}
#'  \item{convergence_tol}{\code{numeric = -Inf} defines early stopping strategy. We stop fitting
#'     when one of two following conditions will be satisfied: (a) we have used
#'     all iterations, or (b) relative change of frobenious norm of the two consequent solution is less then
#'     provided \code{convergence_tol}}
#'  \item{...}{other arguments. Not used at the moment}
#' }
#' @export
PureSVD = R6::R6Class(
  inherit = BaseRecommender,
  classname = "PureSVD",
  public = list(
    initialize = function(rank = 10L,
                          lambda = 0,
                          init = NULL,
                          preprocess = identity,
                          ...) {
      private$rank = rank
      private$lambda = lambda
      private$init = init
      private$set_internal_matrix_formats(sparse = "sparseMatrix", dense = NULL)
      stopifnot(is.function(preprocess))
      private$preprocess = preprocess
    },
    fit_transform = function(x, n_iter = 10L, convergence_tol = 1e-3, ...) {
      x = private$check_convert_input(x)
      x = private$preprocess(x)
      private$item_ids = colnames(x)
      n_user = nrow(x)
      n_item = ncol(x)
      private$svd = soft_svd(x, rank = private$rank,
                     lambda = private$lambda,
                     n_iter = n_iter,
                     convergence_tol = convergence_tol,
                     init = private$init,
                     ...)
      res = private$svd$u %*% diag(x = private$svd$d)
      private$components_ = t(private$svd$v %*%  diag(x = private$svd$d))
      invisible(res)
    },
    transform = function(x, ...) {
      x = private$check_convert_input(x)
      x = private$preprocess(x)
      res = x %*% private$svd$v
      rownames(res) = rownames(x)
      as.matrix(res)
    }
  ),
  private = list(
    rank = NULL,
    lambda = NULL,
    init = NULL,
    svd = NULL,
    preprocess = NULL
  )
)
