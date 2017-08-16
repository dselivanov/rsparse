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
      str(lhs)
      flog.debug("solving least squares")
      self$components = solve(lhs, rhs)
    },
    predict = function(x, k, n_threads = 1L, ...) {
      user_item_scores = x %*% self$orthogonal_basis %*% self$components
      res = top_k_indices_byrow(as.matrix(user_item_scores), x, k, n_threads)
      res
    },
    orthogonal_basis = NULL,
    components = NULL
  ),
  private = list(
    rank = NULL,
    svd_solver = NULL,
    lambda = NULL
  )
)
