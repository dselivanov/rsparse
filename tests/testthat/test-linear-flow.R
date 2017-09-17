futile.logger::flog.threshold(futile.logger::WARN)
train = movielens_100k_ratings[1:900, , drop = F]
cv = movielens_100k_ratings[901:nrow(movielens_100k_ratings), , drop = F]
futile.logger::flog.threshold(futile.logger::INFO)

test_that("test linear-flow", {
  lambda = 0
  rank = 8
  K = 10
  cv_split = split_into_cv(cv)
  model = LinearFlow$new(rank = rank, lambda = lambda,
                         svd_solver = "irlba", Q = NULL)

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
  fit_trace = model$cv(x = train, x_cv_train = cv_split$x_train, x_cv_cv = cv_split$x_cv,
           lambdas = "auto@50", metric = "map@10", n_threads = 4, not_recommend = cv_split$x_train)
  expect_equal(nrow(fit_trace), 50)
  expect_gt(fit_trace$lambda[[2]],  fit_trace$lambda[[1]])
})
