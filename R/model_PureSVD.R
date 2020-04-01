#' @name PureSVD
#'
#' @title PureSVD recommender model decompomposition
#' @description Creates PureSVD recommender model. Solver is based on Soft-SVD which is
#' very similar to truncated SVD but optionally adds regularization based on nuclear norm.
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
    #' @description create PureSVD model
    #' @param rank size of the latent dimension
    #' @param lambda regularization parameter
    #' @param init initialization of item embeddings
    #' @param preprocess \code{identity()} by default. User spectified function which will
    #' be applied to user-item interaction matrix before running matrix factorization
    #' (also applied during inference time before making predictions).
    #' For example we may want to normalize each row of user-item matrix to have 1 norm.
    #' Or apply \code{log1p()} to discount large counts.
    #' @param method type of the solver for initialization of the orthogonal
    #' basis. Original paper uses SVD. See paper for details.
    #' @param ... not used at the moment
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

      stopifnot(is.function(preprocess))
      private$preprocess = preprocess
    },
    #' @description performs matrix factorization
    #' @param x input sparse user-item matrix(of class \code{dgCMatrix})
    #' @param n_iter maximum number of iterations
    #' @param convergence_tol \code{numeric = -Inf} defines early stopping strategy.
    #' Stops fitting when one of two following conditions will be satisfied: (a) passed
    #' all iterations (b) relative change of Frobenious norm of the two consequent solution
    #' is less then provided \code{convergence_tol}.
    #' @param ... not used at the moment
    fit_transform = function(x, n_iter = 100L, convergence_tol = 1e-3, ...) {
      uids = rownames(x)
      private$item_ids = colnames(x)
      x = as(x, "sparseMatrix")
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

      self$components = t(private$svd$v %*%  diag(x = private$svd$d))
      data.table::setattr(self$components, "dimnames", list(NULL, private$item_ids))


      private$components_ = t(private$svd$v)
      data.table::setattr(private$components_, "dimnames", list(NULL, private$item_ids))
      invisible(res)
    },
    #' @description calculates user embeddings for the new input
    #' @param x input matrix
    #' @param ... not used at the moment
    transform = function(x, ...) {
      uids = rownames(x)
      x = as(x, "sparseMatrix")
      x = private$preprocess(x)
      res = x %*% private$svd$v
      res = as.matrix(res)
      data.table::setattr(res, "dimnames", list(uids, NULL))
      res
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
