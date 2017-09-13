#' @name LinearFlow
#'
#' @title Linear Flow method for collaborative filtering
#' @description Creates model which seeks for item similarity matrix
#' @format \code{R6Class} object.
#' @section Usage:
#' For usage details see \bold{Methods, Arguments and Examples} sections.
#' \preformatted{
#'   model = LinearFlow$new( rank = 8L,
#'                           lambda = 0,
#'                           svd_solver = c("irlba", "randomized_svd"),
#'                           Q = NULL)
#'   model$fit_transform(x, ...)
#'   model$predict(x, k, n_threads = 1L, not_recommend = x, ...)
#'   model$components
#'   model$Q
#'   model$cv = function(x, x_cv_train, x_cv_cv, lambdas = "auto@@50", metric = "map@@10",
#'                       n_threads = 1L, not_recommend = x_cv_train, ...)
#' }
#' @export
LinearFlow = R6::R6Class(
  classname = "LinearFlow",
  public = list(
    Q = NULL,
    components = NULL,
    initialize = function(rank = 8L,
                          lambda = 0,
                          svd_solver = c("irlba", "randomized_svd"),
                          Q = NULL
    ) {
      private$rank = rank
      private$svd_solver = match.arg(svd_solver)
      private$lambda = lambda
      self$Q = Q
    },
    fit_transform = function(x, ...) {
      stopifnot(!is.null(colnames(x)))
      private$item_ids = colnames(x)
      self$Q = private$calc_Q(x, ...)
      flog.debug("calculating RHS")

      # rhs = t(self$Q) %*% t(x) %*% x
      # same as above but a bit faster:
      rhs = crossprod(x %*% self$Q, x)

      flog.debug("calculating LHS")
      lhs = rhs %*% self$Q
      self$components = private$fit_transform_internal(lhs, rhs, private$lambda, ...)
      invisible(as.matrix(x %*% self$Q))
    },
    cv = function(x, x_cv_train, x_cv_cv, lambdas = "auto@50", metric = "map@10",
                  n_threads = 1L, not_recommend = x_cv_train, ...) {

      stopifnot(!is.null(colnames(x)))
      private$item_ids = colnames(x)


      stopifnot(private$item_ids == colnames(x_cv_cv))
      stopifnot(private$item_ids == colnames(x_cv_train))

      lambda_auto = FALSE
      if(is.character(lambdas)) {
        if (length(grep(pattern = "(auto)\\@[[:digit:]]+", x = lambdas)) != 1 )
          stop(sprintf("don't know how add '%s' metric 'auto@k' or numeric are supported", lambdas))
        lambdas = strsplit(lambdas, "@", T)[[1]]
        lambdas_k = as.integer(lambdas[[2]])
        lambda_auto = TRUE
      } else {
        stopifnot(is.numeric(lambdas))
      }

      if (length(grep(pattern = "(ndcg|map)\\@[[:digit:]]+", x = metric)) != 1 )
        stop(sprintf("don't know how add '%s' metric. Only 'map@k', 'ndcg@k' are supported", metric))
      metric = strsplit(metric, "@", T)[[1]]
      metric_k = as.integer(metric[[2]])
      metric_name = metric[[1]]

      self$Q = private$calc_Q(x, ...)
      flog.info("calculating RHS")
      # rhs = t(self$Q) %*% t(x) %*% x
      # same as above but a bit faster:
      rhs = crossprod(x %*% self$Q, x)

      flog.info("calculating LHS")
      lhs = rhs %*% self$Q
      # calculate "reasonable" lambda from values of main diagonal of LSH
      if(lambda_auto) {
        lhs_ridge = diag(lhs)
        # generate sequence of lambdas
        lambdas = seq(log10(0.1 * min(lhs_ridge)), log10(100 * max(lhs_ridge)), length.out = lambdas_k)
        lambdas = 10 ^ lambdas
      }

      cv_res = data.frame(lambda = lambdas, score = NA_real_)
      xq_cv_train = as.matrix(x_cv_train %*% self$Q)

      for(i in seq_along(lambdas)) {
        lambda = lambdas[[i]]
        Y = private$fit_transform_internal(lhs, rhs, lambda, ...)
        preds = private$predict_internal(xq_cv_train, k = metric_k, Y = Y, n_threads = n_threads, not_recommend = not_recommend)
        score = NULL
        if(metric_name == "map")
          score = mean(ap_k(preds, x_cv_cv, ...), na.rm = T)
        if(metric_name == "ndcg")
          score = mean(ndcg_k(preds, x_cv_cv, ...), na.rm = T)

        cv_res$score[[i]] = score
        if(score >= max(cv_res$score, na.rm = T) || is.null(self$components)) {
          self$components = Y
          private$lambda = lambda
        }
        flog.info("%d/%d lambda %.3f score = %.3f", i, length(lambdas), lambda, score)
      }
      cv_res
    },
    predict = function(x, k, n_threads = 1L, not_recommend = x, ...) {
      xq = x %*% self$Q
      predicted_item_ids = private$predict_internal(xq, k = k, self$components, n_threads = n_threads, not_recommend = not_recommend, ...)
      predicted_item_ids
    }
  ),
  private = list(
    rank = NULL,
    svd_solver = NULL,
    lambda = NULL,
    item_ids = NULL,
    calc_Q = function(x, ...) {
      result = NULL
      if(!is.null(self$Q)) {
        flog.debug("found Q, checking it...")
        stopifnot(nrow((self$Q)) == ncol(x))
        stopifnot(ncol((self$Q)) == private$rank)
        result = self$Q
      } else {
        if(is.null(self$Q)) {
          if(private$svd_solver == "irlba") {
            flog.info("fitting truncated SVD with irlba")
            trunc_svd = irlba::irlba(x, nv = private$rank, tol = 1e-4)
          } else {
            if(private$svd_solver == "randomized_svd") {
              flog.info("fitting truncated SVD with randomized algorithm")
              trunc_svd = irlba::svdr(x, private$rank)
            } else
                stop(sprintf("don't know %s", private$svd_solver))
          }
        }
        result = trunc_svd$v
      }
      stopifnot(is.numeric(result))
      result
    },
    fit_transform_internal = function(lsh, rhs, lambda, ...) {
      flog.debug("solving least squares with lambda %.3f", lambda)
      lhs_ridge = lsh + Diagonal(private$rank, lambda)
      as.matrix(solve(lhs_ridge, rhs))
    },
    predict_internal = function(xq, k, Y, n_threads = 1L, not_recommend = x, ...) {
      if(!is.matrix(xq))
        xq = as.matrix(xq)
      if(!is.matrix(Y))
        Y = as.matrix(Y)

      flog.debug("predicting top %d values", k)
      indices = dotprod_top_k(xq, Y, k, n_threads, not_recommend)

      scores = attr(indices, "scores", exact = TRUE)
      data.table::setattr(indices, "scores", NULL)

      # predicted_item_ids = colnames(x)[indices]
      predicted_item_ids = private$item_ids[indices]
      data.table::setattr(predicted_item_ids, "dim", dim(indices))
      data.table::setattr(predicted_item_ids, "indices", indices)
      data.table::setattr(predicted_item_ids, "scores", scores)
      data.table::setattr(predicted_item_ids, "dimnames", list(rownames(xq), NULL))
      predicted_item_ids
    }
  )
)
