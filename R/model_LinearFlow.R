#' @name LinearFlow
#'
#' @title Linear-FLow method for one-class collaborative filtering
#' @description Creates \bold{Linear-FLow} model described in
#' \href{http://www.bkveton.com/docs/ijcai2016.pdf}{Practical Linear Models for Large-Scale One-Class Collaborative Filtering}.
#' The goal is to find item-item (or user-user) similarity matrix which is \bold{low-rank and has small Frobenius norm}. Such
#' double regularization allows to better control the generalization error of the model.
#' Idea of the method is somewhat similar to \bold{Sparse Linear Methods(SLIM)} but scales to large datasets much better.
#' @seealso
#' \itemize{
#'   \item{\url{http://www.bkveton.com/docs/ijcai2016.pdf}}
#'   \item{\url{http://www-users.cs.umn.edu/~xning/slides/ICDM2011_slides.pdf}}
#' }
#' @format \code{R6Class} object.
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#'   model = LinearFlow$new( rank = 8L,
#'                           lambda = 0,
#'                           solve_right_singular_vectors = c("soft_impute", "svd"),
#'                           n_threads = parallel::detectCores(),
#'                           v = NULL,
#'                           preprocess = identity,
#'                           ...)
#'   model$fit_transform(x, ...)
#'   model$transform(x, ...)
#'   model$predict(x, k, not_recommend = x, ...)
#'   model$components
#'   model$v
#'   model$cross_validate_lambda(x, x_train, x_test, lambda = "auto@@10",
#'                        metric = "map@@10", not_recommend = x_train, ...)
#' }
#' @format \code{R6Class} object.
#' @section Usage:
#' @section Methods:
#' \describe{
#'   \item{\code{$new(rank = 8L, lambda = 0,
#'               solve_right_singular_vectors = c("svd", "soft_impute"),
#'               n_threads = parallel::detectCores(),
#'               v = NULL, preprocess = identity, ...)}}{ creates Linear-FLow model with \code{rank} latent factors.
#'     If \code{v} (right singular vectors of the user-item interactions matrix)
#'     is provided then model initialized with its values.}
#'   \item{\code{$fit_transform(x, ...)}}{ fits model to
#'     an input user-item matrix.
#'     \bold{Returns factor matrix for users of size \code{n_users * rank}}}
#'   \item{\code{$transform(x, ...)}}{transforms (new) sparse user-item interaction matrix into user-embeddings matrix.}
#'   \item{\code{$predict(x, k, not_recommend = x, ...)}}{predict \bold{top k}
#'     item ids for users \code{x}. Users features should be defined the same way as they were defined in
#'     training data - as \bold{sparse matrix}. Column names (=item ids) should be in the same order as
#'     in the \code{fit_transform()}.}
#'   \item{preprocess}{\code{function} = \code{identity()} by default. User spectified function which will
#'     be applied to user-item interaction matrix before running matrix factorization
#'     (also applied in inference time before making predictions).}
#'   \item{\code{$cross_validate_lambda(x, x_train, x_test, lambda = "auto@@10", metric = "map@@10",
#'                               not_recommend = x_train, ...)}}{perfroms search of the
#'   best regularization parameter \code{lambda}:
#'   \enumerate{
#'     \item Model is trained on \code{x} data
#'     \item Then model makes predictions based on \code{x_train} data
#'     \item And finally these predications are validated using specified \code{metric} against \code{x_test} data
#'   }
#'   Note that this is implemented smartly with \bold{"warm starts"}.
#'   So it is very cheap - \bold{cost is almost the same as for single fit} of the model. The only considerable additional cost is
#'   time to predict \emph{top k} items. In most cases automatic lambda like \code{lambda = "auto@@20"} is able to find good value of the parameter}
#'   \item{\code{$components}}{item factors matrix of size \code{rank * n_items}. In the paper this matrix is called \bold{Y}}
#'   \item{\code{$v}}{right singular vector of the user-item matrix. Size is \code{n_items * rank}. In the paper this matrix is called \bold{v}}
#'}
#' @section Arguments:
#' \describe{
#'  \item{model}{A \code{LinearFlow} model.}
#'  \item{x}{An input sparse user-item matrix (inherits from \code{sparseMatrix})}
#'  \item{rank}{\code{integer} - number of latent factors}
#'  \item{lambda}{\code{numeric} - regularization parameter or sequence of regularization values for \code{cross_validate_lambda} method.}
#'  \item{n_threads}{\code{numeric} default number of threads to use during prediction (if OpenMP is available).
#'  At the training most expensive stage is truncated SVD calculation. \code{svd} method on \code{dgCMatrix} relies on system BLAS,
#'  so it also can benefit from multithreded BLAS. But this is not controlled by \code{n_threads} parameter.
#'  For changing number of BLAS threads at runtime please check \href{https://cran.r-project.org/package=RhpcBLASctl}{RhpcBLASctl package}.}
#'  \item{not_recommend}{\code{sparse matrix} or \code{NULL} - points which items should be excluided from recommendations for a user.
#'    By default it excludes previously seen/consumed items.}
#'  \item{metric}{metric to use in evaluation of top-k recommendations.
#'    Currently only \code{map@@k} and \code{ndcg@@k} are supported (\code{k} can be any integer).}
#'  \item{...}{other arguments (not used at the moment)}
#' }
#' @export
LinearFlow = R6::R6Class(
  classname = "LinearFlow",
  inherit = mlapi::mlapiDecomposition,
  public = list(
    v = NULL,
    n_threads = NULL,
    initialize = function(rank = 8L,
                          lambda = 0,
                          solve_right_singular_vectors = c("soft_impute", "svd"),
                          n_threads = parallel::detectCores(),
                          v = NULL,
                          preprocess = identity) {
      self$n_threads = n_threads
      private$preprocess = preprocess
      private$rank = as.integer(rank)
      private$solve_right_singular_vectors = match.arg(solve_right_singular_vectors)
      private$lambda = as.numeric(lambda)
      self$v = v
    },
    fit_transform = function(x, ...) {
      stopifnot(inherits(x, "sparseMatrix") || inherits(x, "SparsePlusLowRank"))
      x = private$preprocess(x)
      private$item_ids = colnames(x)
      self$v = private$get_right_singular_vectors(x, ...)
      flog.debug("calculating RHS")

      # rhs = t(self$v) %*% t(x) %*% x
      # same as above but a bit faster:
      rhs = crossprod(x %*% self$v, x)

      flog.debug("calculating LHS")
      lhs = rhs %*% self$v
      private$components_ = private$fit_transform_internal(lhs, rhs, private$lambda, ...)
      invisible(as.matrix(x %*% self$v))
    },
    transform = function(x, ...) {
      stopifnot(inherits(x, "sparseMatrix") || inherits(x, "SparsePlusLowRank"))
      x = private$preprocess(x)
      res = x %*% self$v
      if(!is.matrix(res))
        res = as.matrix(res)
      invisible(res)
    },
    cross_validate_lambda = function(x, x_train, x_test, lambda = "auto@10", metric = "map@10",
                  not_recommend = x_train, ...) {

      private$item_ids = colnames(x)


      stopifnot(private$item_ids == colnames(x_test))
      stopifnot(private$item_ids == colnames(x_train))

      x = private$preprocess(x)
      x_train = private$preprocess(x_train)
      x_test = private$preprocess(x_test)

      lambda_auto = FALSE
      if(is.character(lambda)) {
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
      flog.debug("calculating RHS")
      # rhs = t(self$v) %*% t(x) %*% x
      # same as above but a bit faster:
      rhs = crossprod(x %*% self$v, x)

      flog.debug("calculating LHS")
      lhs = rhs %*% self$v
      # calculate "reasonable" lambda from values of main diagonal of LHS
      if(lambda_auto) {
        lhs_ridge = diag(lhs)
        # generate sequence of lambda
        lambda = seq(log10(0.1 * min(lhs_ridge)), log10(10 * max(lhs_ridge)), length.out = lambdas_k)
        lambda = 10 ^ lambda
      }

      cv_res = data.frame(lambda = lambda, score = NA_real_)
      xq_cv_train = as.matrix(x_train %*% self$v)

      for(i in seq_along(lambda)) {
        lambda_i = lambda[[i]]
        Y = private$fit_transform_internal(lhs, rhs, lambda_i, ...)
        preds = private$predict_internal(xq_cv_train, k = metric_k, Y = Y, not_recommend = not_recommend)
        score = NULL
        if(metric_name == "map")
          score = mean(ap_k(preds, x_test, ...), na.rm = T)
        if(metric_name == "ndcg")
          score = mean(ndcg_k(preds, x_test, ...), na.rm = T)

        cv_res$score[[i]] = score
        if(score >= max(cv_res$score, na.rm = T) || is.null(private$components_)) {
          private$components_ = Y
          private$lambda = lambda_i
        }
        flog.info("%d/%d lambda %.3f score = %.3f", i, length(lambda), lambda_i, score)
      }
      cv_res
    },
    predict = function(x, k, not_recommend = x, ...) {
      xq = x %*% self$v
      predicted_item_ids = private$predict_internal(xq, k = k, private$components_,
                                                    not_recommend = not_recommend, ...)
      predicted_item_ids
    }
  ),
  private = list(
    rank = NULL,
    preprocess = NULL,
    solve_right_singular_vectors = NULL,
    lambda = NULL,
    item_ids = NULL,
    get_right_singular_vectors = function(x, ...) {
      result = NULL
      if(!is.null(self$v)) {
        flog.debug("found v, checking it...")
        stopifnot(nrow((self$v)) == ncol(x))
        stopifnot(ncol((self$v)) == private$rank)
        result = self$v
      } else {
        if(is.null(self$v)) {
          if(private$solve_right_singular_vectors == "soft_impute")
            trunc_svd = soft_impute(x, rank = private$rank, lambda = 0, ...)
          else if(private$solve_right_singular_vectors == "svd")
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
      flog.debug("solving least squares with lambda %.3f", lambda)
      lhs_ridge = lhs + diag(rep(lambda, private$rank))
      as.matrix(solve(lhs_ridge, rhs))
    },
    predict_internal = function(xq, k, Y, not_recommend = x, ...) {
      if(!is.matrix(xq))
        xq = as.matrix(xq)
      if(!is.matrix(Y))
        Y = as.matrix(Y)

      flog.debug("predicting top %d values", k)
      indices = dotprod_top_k(xq, Y, k, self$n_threads, not_recommend)
      data.table::setattr(indices, "dimnames", list(rownames(xq), NULL))
      data.table::setattr(indices, "indices", NULL)

      if(!is.null(private$item_ids)) {
        predicted_item_ids = private$item_ids[indices]
        data.table::setattr(predicted_item_ids, "dim", dim(indices))
        data.table::setattr(predicted_item_ids, "dimnames", list(rownames(xq), NULL))
        data.table::setattr(indices, "indices", predicted_item_ids)
      }

      indices
    }
  )
)
