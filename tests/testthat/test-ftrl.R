context("FTRL")

N_SMPL = 5e3
N_FEAT = 1e3
NNZ = N_SMPL * 30

set.seed(1)
i = sample(N_SMPL, NNZ, TRUE)
j = sample(N_FEAT, NNZ, TRUE)
y = sample(c(0, 1), N_SMPL, TRUE)
x = sample(c(-1, 1), NNZ, TRUE)
odd = seq(1, 99, 2)
x[i %in% which(y == 1) & j %in% odd] = 1
m = sparseMatrix(i = i, j = j, x = x, dims = c(N_SMPL, N_FEAT), giveCsparse = FALSE)
x = as(m, "RsparseMatrix")


test_that("FTRL coefficients", {
  ftrl = FTRL$new(learning_rate = 0.01, learning_rate_decay = 0.1, lambda = 20, l1_ratio = 1, dropout = 0)
  ftrl$partial_fit(x, y, nthread = 1)
  w = ftrl$coef()
  expect_equal(sum(sign(w[odd])), 50)
})

test_that("FTRL dump immutable", {
  ftrl = FTRL$new(learning_rate = 0.01, learning_rate_decay = 0.1, lambda = 20, l1_ratio = 1, dropout = 0)
  ftrl$partial_fit(x, y, nthread = 1)
  dump_1 = ftrl$.__enclos_env__$private$dump()
  # make sure dump_1 cloned during dump
  ftrl$partial_fit(x, y, nthread = 1)
  dump_2 = ftrl$.__enclos_env__$private$dump()
  expect_true(any(dump_1$z != dump_2$z))
  expect_true(any(dump_1$n != dump_2$n))

  ftrl2 = FTRL$new()
  ftrl2$.__enclos_env__$private$load(dump_2)
  ftrl2$partial_fit(x, y, nthread = 1)
  dump_3 = ftrl2$.__enclos_env__$private$dump()
  # make sure dump_2 cloned internally and not modified by `partial_fit()` call
  expect_true(any(dump_2$n != dump_3$n))
})

# make sure that we can overfit - 2 passes over data will have better accuracy
test_that("FTRL incremental fit", {
  ftrl = FTRL$new(learning_rate = 0.1, learning_rate_decay = 0.1, lambda = 0.001, l1_ratio = 1, dropout = 0)
  ftrl$partial_fit(x, y, nthread = 1)
  accuracy_1 = sum(ftrl$predict(x, nthread = 1) >= 0.5 & y) / length(y)
  # glmnet::auc(y = y, prob = ftrl$predict(x, nthread = 1))

  ftrl$partial_fit(x, y, nthread = 1)
  accuracy_2 = sum(ftrl$predict(x, nthread = 1) >= 0.5 & y) / length(y)
  expect_gt(accuracy_2, accuracy_1)
})

test_that("FTRL check dimesnions", {
  ftrl = FTRL$new(learning_rate = 0.1, learning_rate_decay = 0.1, lambda = 0.001, l1_ratio = 1, dropout = 0)
  ftrl$partial_fit(x, y, nthread = 1)
  # partial_fit of the model wich initialized with another number of features
  expect_error(ftrl$partial_fit(x[, -1], y, nthread = 1))
  expect_error(ftrl$partial_fit(cbind(x, rep(1, nrow(x))), y, nthread = 1))
  # predict of the model wich initialized with another number of features
  expect_error(ftrl$predict(x[, -1], nthread = 1))
  expect_error(ftrl$predict(cbind(x, rep(1, nrow(x))), nthread = 1))
})
