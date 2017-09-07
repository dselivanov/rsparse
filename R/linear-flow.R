#' @export
LinearFlow = R6::R6Class(
  # inherit = mlapi::mlapiDecomposition,
  classname = "LinearFlow",
  public = list(
    initialize = function(rank = 8L,
                          lambda = 0,
                          svd_solver = c("irlba", "randomized_svd"),
                          orthogonal_basis = NULL
    ) {
      private$rank = rank
      private$svd_solver = match.arg(svd_solver)
      private$lambda = lambda
      self$orthogonal_basis = orthogonal_basis
    },
    fit_transform = function(x, ...) {

      private$item_ids = colnames(x)

      if(!is.null(self$orthogonal_basis)) {
        stopifnot(nrow((self$orthogonal_basis)) == ncol(x))
        stopifnot(ncol((self$orthogonal_basis)) == private$rank)
      } else
      if(is.null(self$orthogonal_basis)) {
        if(private$svd_solver == "irlba") {
          flog.debug("fitting truncated SVD with irlba")
          trunc_svd = irlba::irlba(x, nv = private$rank, tol = 1e-4)
        }
        if(private$svd_solver == "randomized_svd") {
          flog.debug("fitting truncated SVD with randomized algorithm")
          trunc_svd = irlba::svdr(x, private$rank)
        }
        self$orthogonal_basis = trunc_svd$v
      } else
        stop(sprintf("don't know %s", private$svd_solver))
      flog.debug("calculating RHS")
      rhs = t(self$orthogonal_basis) %*% t(x) %*% x
      flog.debug("calculating LHS")
      lhs = rhs %*% self$orthogonal_basis + Diagonal(private$rank, private$lambda)
      flog.debug("solving least squares")
      self$components = solve(lhs, rhs)
    },
    predict = function(x, k, n_threads = 1L, not_recommend = x, ...) {

      user_item_score = x %*% self$orthogonal_basis %*% self$components
      user_item_score = as.matrix(user_item_score)
      indices = top_k_indices_byrow(user_item_score, not_recommend, k, n_threads)
      scores = attr(indices, "scores", exact = TRUE)
      attr(indices, "scores") = NULL
      predicted_item_ids = private$item_ids[indices]
      data.table::setattr(predicted_item_ids, "dim", dim(indices))
      data.table::setattr(predicted_item_ids, "indices", indices)
      data.table::setattr(predicted_item_ids, "scores", scores)
      data.table::setattr(predicted_item_ids, "dimnames", list(rownames(x), NULL))
      predicted_item_ids
    },
    orthogonal_basis = NULL,
    components = NULL
  ),
  private = list(
    rank = NULL,
    svd_solver = NULL,
    lambda = NULL,
    item_ids = NULL
  )
)
