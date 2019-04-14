context("PureSVD")

logger = lgr::get_logger('rsparse')
logger$set_threshold('warn')

train = movielens100k[1:900, ]
cv = movielens100k[901:nrow(movielens100k), ]

test_that("test PureSVD", {
  rank = 10
  lambda = 0
  model = PureSVD$new(rank = rank,  lambda = lambda)
  user_emb = model$fit_transform(train, n_iter = 20, convergence_tol = 0.001)
  # check dimensions
  expect_equal(dim(user_emb), c(nrow(train), rank))
  expect_equal(rownames(user_emb), rownames(train))
  # check it predicts
  N = 10
  preds = model$predict(cv, N)
  expect_equal(rownames(preds), rownames(cv))
  expect_equal(dim(preds), c(nrow(cv), N))
  user_emb = model$transform(cv)
  expect_equal(dim(user_emb), c(nrow(cv), rank))
  expect_equal(colnames(model$components), colnames(train))
})
