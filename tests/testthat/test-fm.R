context("FM")

test_that("test FM can preidict XOR function", {
  x = rbind(
    c(0, 0),
    c(0, 1),
    c(1, 0),
    c(1, 1)
  )
  y = c(0, 1, 1, 0)

  x = as(x, "RsparseMatrix")
  fm = FactorizationMachine$new(learning_rate_w = 10, rank = 2, lambda_w = 0, lambda_v = 0, family = 'binomial', intercept = TRUE)
  res = fm$fit(x, y, n_iter = 100)
  preds = fm$predict(x)
  expect_true(all(preds[c(1, 4)] < 0.01))
  expect_true(all(preds[c(2, 3)] > 0.99))
})
