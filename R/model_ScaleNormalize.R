#' @export
ScaleNormalize = R6::R6Class(
  public = list(
    norm = NULL,
    scale = NULL,
    scaling_matrix = NULL,
    initialize = function(scale = 0.5, norm = 2, target = c("rows", "columns")) {
      self$norm = norm
      self$scale = scale

      target = match.arg(target)
      private$target = target

      private$FUN = if (target == "rows") {
        rowSums
      } else {
        colSums
      }
    },
    fit = function(x) {
      norm_vec = private$FUN(x ^ self$norm) ** (1 / self$norm)
      #norm_vec[is.infinite(norm_vec)] = 0
      i = norm_vec != 0
      norm_vec[i] = norm_vec[i] ** (self$scale - 1)
      self$scaling_matrix = Diagonal(x = norm_vec)

    },
    transform = function(x) {
      if(private$target == "rows") {
        res = self$scaling_matrix %*% x
      } else {
        res = x %*% self$scaling_matrix
      }
      res
    },
    fit_transform = function(x) {
      self$fit(x)
      self$transform(x)
    }
  ),
  private = list(
    FUN = NULL,
    target = NULL
  )
)
