#' @title Re-scales input matrix proportinally to item popularity
#' @description scales input user-item interaction matrix as per eq (16) from the paper.
#' Usage of such rescaled matrix with [PureSVD] model will be equal to running PureSVD
#' on the scaled cosine-based inter-item similarity matrix.
#' @references See \href{https://arxiv.org/pdf/1511.06033.pdf}{EigenRec: Generalizing PureSVD for
#' Effective and Efficient Top-N Recommendations} for details.
#' @export
ScaleNormalize = R6::R6Class(
  public = list(
    #' @field norm which norm model should make equal to one
    norm = NULL,
    #' @field scale how to rescale norm vector
    scale = NULL,
    #' @description creates model
    #' @param scale numeric, how to rescale norm vector
    #' @param norm numeric, which norm model should make equal to one
    #' @param target character, defines whether rows or columns should be rescaled
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
    #' @description fits the modes
    #' @param x input sparse matrix
    fit = function(x) {
      norm_vec = private$FUN(x ^ self$norm) ** (1 / self$norm)
      #norm_vec[is.infinite(norm_vec)] = 0
      i = norm_vec != 0
      norm_vec[i] = norm_vec[i] ** (self$scale - 1)
      private$scaling_matrix = Diagonal(x = norm_vec)

    },
    #' @description transforms new matrix
    #' @param x input sparse matrix
    transform = function(x) {
      if (private$target == "rows") {
        res = private$scaling_matrix %*% x
      } else {
        res = x %*% private$scaling_matrix
      }
      res
    },
    #' @description fits the model and transforms input
    #' @param x input sparse matrix
    fit_transform = function(x) {
      self$fit(x)
      self$transform(x)
    }
  ),
  private = list(
    FUN = NULL,
    scaling_matrix = NULL,
    target = NULL
  )
)
