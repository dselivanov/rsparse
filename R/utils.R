#' @name train_test_split
#' @title Creates cross-validation set from user-item interactions
#' @description Basic splitting of the user-item interaction matrix into train and testing part.
#' Useful for when data doesn't have time dimension.
#' Usually during model tuning it worth to keep some \code{x} matrix as hold-out data set.
#' Then this \code{x} could be splitted in 2 parts - \emph{train} and \emph{test}.
#' Model tries to predict \emph{test} data using \emph{train}
#' @param x sparse user-item interation matrix. Internally \code{Matrix::TsparseMatrix} is used.
#' @param test_proportion - proportion of the observations for each user to keep as "test" data.
#' @keywords internal
train_test_split = function(x, test_proportion = 0.5) {
  stopifnot(inherits(x, "sparseMatrix"))
  temp = as(x, "TsparseMatrix")
  cv_proportion = 1 - test_proportion
  # make R CMD check happy (avoid "no visible binding for global variable" warnings)
  i = train = NULL
  temp = data.table(i = temp@i, j = temp@j, x = temp@x)
  temp[, train := sample(c(FALSE, TRUE), .N, replace = TRUE, prob = c(test_proportion, cv_proportion)), keyby = i]
  x_train = temp[train == TRUE]
  x_test = temp[train == FALSE]
  rm(temp)

  x_train = sparseMatrix( i = x_train$i, j = x_train$j, x = x_train$x,
                          dims = dim(x), dimnames = dimnames(x), index1 = FALSE)
  x_test = sparseMatrix( i = x_test$i, j = x_test$j, x = x_test$x,
                              dims = dim(x), dimnames = dimnames(x), index1 = FALSE)
  list(train = x_train, test = x_test)
}


find_top_product = function(x, y, k, not_recommend = NULL, exclude = integer(0), n_threads = getOption("rsparse_omp_threads", 1L)) {
  n_threads_blas = RhpcBLASctl::blas_get_num_procs()
  # set num threads to 1 in order to avoid thread contention between BLAS and openmp threads in `top_product()`
  RhpcBLASctl::blas_set_num_threads(1L)
  # restore on exit
  on.exit(RhpcBLASctl::blas_set_num_threads(n_threads_blas))

  if (!inherits(exclude, "integer"))
    stop("'exclude' should be integer vector")
  if (!(is.null(not_recommend) || inherits(not_recommend, "sparseMatrix")))
    stop("'not_recommend' should be NULL or 'sparseMatrix'")

  stopifnot(ncol(x) == nrow(y))

  if (is.null(not_recommend))
    not_recommend = new("dgRMatrix")
  else {
    stopifnot(nrow(x) == nrow(not_recommend))
    stopifnot(ncol(y) == ncol(not_recommend))
    not_recommend = as(not_recommend, "RsparseMatrix")
  }
  top_product(x, y, k, n_threads, not_recommend, exclude)
}
