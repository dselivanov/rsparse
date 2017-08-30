#' @export
split_into_cv = function(x, train_proportion = 0.5) {
  stopifnot(inherits(x, "sparseMatrix"))
  temp = as(x, "TsparseMatrix")
  cv_proportion = 1 - train_proportion

  temp = data.table(i = temp@i, j = temp@j, x = temp@x)
  temp[, train := sample(c(TRUE, FALSE), .N, replace = TRUE, prob = c(train_proportion, cv_proportion)), by = i]
  x_train = temp[train == TRUE]
  x_cv = temp[train == FALSE]
  rm(temp)
  x_train = sparseMatrix( i = x_train$i, j = x_train$j, x = x_train$x,
                          dims = dim(x), dimnames = dimnames(x), index1 = FALSE)
  x_cv = sparseMatrix( i = x_cv$i, j = x_cv$j, x = x_cv$x,
                              dims = dim(x), dimnames = dimnames(x), index1 = FALSE)
  list(x_train = x_train, x_cv = x_cv)
}
