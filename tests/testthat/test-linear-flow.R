context("LinearFlow")

logger = lgr::get_logger('rsparse')
logger$set_threshold('warn')

train = movielens100k[1:900, ]
cv = movielens100k[901:nrow(movielens100k), ]

test_that("test linear-flow", {
  lambda = 0
  rank = 8
  K = 10
  cv_split = rsparse:::train_test_split(cv)
  model = LinearFlow$new(rank = rank, lambda = lambda,
                         init = NULL,
                         solve_right_singular_vectors = "svd")

  user_emb = model$fit_transform(train)
  expect_equal(dim(user_emb), c(nrow(train), rank))
  expect_equal(rownames(user_emb), rownames(train))

  expect_equal(colnames(model$components), colnames(train))

  preds = model$predict(cv, k = K)
  expect_equal(rownames(preds), rownames(cv))
  expect_equal(dim(preds), c(nrow(cv), K))

  user_emb = model$transform(cv)
  expect_equal(dim(user_emb), c(nrow(cv), rank))

  # check cheap cross-validation
  fit_trace = model$cross_validate_lambda(x = train, x_train = cv_split$train, x_test = cv_split$test,
           lambda = "auto@50", metric = "map@10", not_recommend = cv_split$train)
  expect_equal(nrow(fit_trace), 50)
  expect_gt(fit_trace$lambda[[2]],  fit_trace$lambda[[1]])
})
