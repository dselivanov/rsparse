#' @name train_test_split
#' @title Creates cross-validation set from user-item interactions
#' @description Basic splitting of the user-item interaction matrix into train and testing part.
#' Useful for when data doesn't have time dimension.
#' Usually during model tuning it worth to keep some \code{x} matrix as hold-out data set.
#' Then this \code{x} could be splitted in 2 parts - \emph{train} and \emph{test}.
#' Model tries to predict \emph{test} data using \emph{train}
#' @param x sparse user-item interation matrix. Internally \code{Matrix::TsparseMatrix} is used.
#' @param test_proportion - proportion of the observations for each user to keep as "test" data.
#' @export
train_test_split = function(x, test_proportion = 0.5) {
  stopifnot(inherits(x, "sparseMatrix"))
  temp = as(x, "TsparseMatrix")
  cv_proportion = 1 - test_proportion
  # make R CMD check happy (avoid "no visible binding for global variable" warnings)
  i = train = NULL
  temp = data.table(i = temp@i, j = temp@j, x = temp@x)
  temp[, train := sample(c(FALSE, TRUE), .N, replace = TRUE, prob = c(test_proportion, cv_proportion)), by = i]
  x_train = temp[train == TRUE]
  x_cv = temp[train == FALSE]
  rm(temp)
  x_train = sparseMatrix( i = x_train$i, j = x_train$j, x = x_train$x,
                          dims = dim(x), dimnames = dimnames(x), index1 = FALSE)
  x_cv = sparseMatrix( i = x_cv$i, j = x_cv$j, x = x_cv$x,
                              dims = dim(x), dimnames = dimnames(x), index1 = FALSE)
  list(x_train = x_train, x_cv = x_cv)
}
