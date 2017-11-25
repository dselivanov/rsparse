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
#'                       n_threads = parallel::detectCores(),
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
#'                    n_threads = parallel::detectCores(),
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
#'   \item{n_threads}{\code{numeric} default number of threads to use during training and prediction
#'   (if OpenMP is available).}
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
#'  \item{n_threads}{\code{numeric} default number of threads to use during training and prediction
#'  (if OpenMP is available).}
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
  inherit = mlapi::mlapiDecomposition,
  classname = "PureSVD",
  public = list(
    n_threads = NULL,
    initialize = function(rank = 10L,
                          lambda = 0,
                          n_threads = parallel::detectCores(),
                          init = NULL,
                          preprocess = identity,
                          ...) {
      private$rank = rank
      private$lambda = lambda
      private$init = init
      self$n_threads = n_threads
      private$set_internal_matrix_formats(sparse = "sparseMatrix", dense = NULL)
      stopifnot(is.function(preprocess))
      private$preprocess = preprocess
    },
    fit_transform = function(x, n_iter = 10, convergence_tol = 1e-3, ...) {
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
      private$components_ = t(private$svd$v * sqrt(private$svd$d))
      res = private$svd$u * sqrt(private$svd$d)
      as.matrix(res)
    },
    transform = function(x, ...) {
      x = private$check_convert_input(x)
      x = private$preprocess(x)
      res = solve_iter_als_svd(x = x, svd_current = private$svd, lambda = private$lambda, singular_vectors = "v")
      as.matrix(res)
    },
    predict = function(x, k, not_recommend = x, ...) {
      stopifnot(private$item_ids == colnames(x))
      stopifnot(is.null(not_recommend) || inherits(not_recommend, "sparseMatrix"))
      if(!is.null(not_recommend))
        not_recommend = as(not_recommend, "dgCMatrix")
      m = nrow(x)

      # transform user features into latent space
      # calculate scores for each item
      # user_item_score = self$transform(x) %*% private$components_
      indices = dotprod_top_k(self$transform(x), private$components_, k, self$n_threads, not_recommend)
      data.table::setattr(indices, "dimnames", list(rownames(x), NULL))
      data.table::setattr(indices, "indices", NULL)

      if(!is.null(private$item_ids)) {
        predicted_item_ids = private$item_ids[indices]
        data.table::setattr(predicted_item_ids, "dim", dim(indices))
        data.table::setattr(predicted_item_ids, "dimnames", list(rownames(x), NULL))
        data.table::setattr(indices, "indices", predicted_item_ids)
      }
      indices
    }
  ),
  private = list(
    item_ids = NULL,
    rank = NULL,
    lambda = NULL,
    init = NULL,
    svd = NULL
  )
)
