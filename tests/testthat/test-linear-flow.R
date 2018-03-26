context("LinearFlow")

futile.logger::flog.threshold(futile.logger::WARN)
train = movielens100k[1:900, , drop = F]
cv = movielens100k[901:nrow(movielens100k), , drop = F]

test_that("test linear-flow", {
  lambda = 0
  rank = 8
  K = 10
  cv_split = train_test_split(cv)
  model = LinearFlow$new(rank = rank, lambda = lambda,
                         solve_right_singular_vectors = "svd", v = NULL)

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
  fit_trace = model$cross_validate_lambda(x = train, x_train = cv_split$x_train, x_test = cv_split$x_cv,
           lambda = "auto@50", metric = "map@10", n_threads = 4, not_recommend = cv_split$x_train)
  expect_equal(nrow(fit_trace), 50)
  expect_gt(fit_trace$lambda[[2]],  fit_trace$lambda[[1]])
})
