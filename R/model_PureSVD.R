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
#'   model$transform(x, ...)
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
#'   \item{\code{$transform(x, ...)}}{Calculates user embeddings from given \code{x} user-item matrix.
#'     Result is \code{n_users * rank} matrix}
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
#' @examples
#' data('movielens100k')
#' i_train = sample(nrow(movielens100k), 900)
#' i_test = setdiff(seq_len(nrow(movielens100k)), i_train)
#' train = movielens100k[i_train, ]
#' test = movielens100k[i_test, ]
#' rank = 32
#' lambda = 0
#' model = PureSVD$new(rank = rank,  lambda = lambda)
#' user_emb = model$fit_transform(sign(test), n_iter = 100, convergence_tol = 0.00001)
#' item_emb = model$components
#' preds = model$predict(sign(test), k = 1500, not_recommend = NULL)
#' mean(ap_k(preds, actual = test))
PureSVD = R6::R6Class(
  inherit = MatrixFactorizationRecommender,
  classname = "PureSVD",
  public = list(
    initialize = function(rank = 10L,
                          lambda = 0,
                          init = NULL,
                          preprocess = identity,
                          method = c('svd', 'impute'),
                          ...) {
      private$rank = rank
      private$lambda = lambda
      private$init = init
      private$method = match.arg(method)

      private$set_internal_matrix_formats(sparse = "sparseMatrix", dense = NULL)
      stopifnot(is.function(preprocess))
      private$preprocess = preprocess
    },
    fit_transform = function(x, n_iter = 100L, convergence_tol = 1e-3, ...) {
      uids = rownames(x)
      private$item_ids = colnames(x)

      x = private$check_convert_input(x)
      x = private$preprocess(x)

      if (private$method == "svd") {
        FUN = soft_svd
      } else {
        FUN = soft_impute
      }

      private$svd = FUN(x, rank = private$rank,
                             lambda = private$lambda,
                             n_iter = n_iter,
                             convergence_tol = convergence_tol,
                             init = private$init,
                             ...)

      res = as.matrix(x %*% private$svd$v)
      data.table::setattr(res, "dimnames", list(uids, NULL))

      #private$components_ = t(private$svd$v %*%  diag(x = private$svd$d))
      private$components_ = t(private$svd$v)
      data.table::setattr(private$components_, "dimnames", list(NULL, private$item_ids))
      invisible(res)
    },
    transform = function(x, ...) {
      uids = rownames(x)
      x = private$check_convert_input(x)
      x = private$preprocess(x)
      res = x %*% private$svd$v
      res = as.matrix(res)
      data.table::setattr(res, "dimnames", list(uids, NULL))
      res
    }
  ),
  active = list(
    components = function(value) {
      if (!missing(value))
        stop("Sorry this is a read-only variable.")
      else {
        if (is.null(private$components_)) {
          warning("Decomposition model was not fitted yet!")
          NULL
        }
        else t(private$svd$v %*%  diag(x = private$svd$d))
      }
    }
  ),
  private = list(
    rank = NULL,
    lambda = NULL,
    init = NULL,
    svd = NULL,
    preprocess = NULL,
    method = NULL
  )
)
