context("BPR-MF")

test_that("BPR-MF", {

  data("movielens100k")
  x = as(movielens100k, "RsparseMatrix")
  lambda_user = lambda_item_positive = lambda_item_negative = 0.001

  rank = 8

  n_users = nrow(x)
  n_items = ncol(x)

  set.seed(1)
  W = matrix(runif(rank * n_users, 0, 1), ncol = n_users)
  H = matrix(runif(rank * n_items, 0, 1), ncol = n_items)

  learning_rate = 0.1
  momentum = 0
  n_updates = n_users * 100
  thresh = 0.5
  n_threads = 1
  update_items = TRUE

  # optimization criterion
  BPR = 0
  WARP = 1
  # link functions
  DOT_PRODUCT = 0
  LOGISTIC = 1

  res = capture.output(rsparse:::warp_solver_double(x, W, H, rank, n_updates, learning_rate, momentum, lambda_user, lambda_item_positive, lambda_item_negative, n_threads, update_items, BPR, DOT_PRODUCT))
  res = fread(text = paste(res, collapse = "\n"), header = FALSE, col.names = c("progress", "AUC", 'oversampling'))
  expect_equal(res[.N, AUC], "AUC:0.816")
})

