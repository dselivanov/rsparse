context("GloVe")

logger = lgr::get_logger('rsparse')
logger$set_threshold('warn')


k = 10
tcm = crossprod(sign(movielens100k))

test_that("test GloVe", {
  model = GloVe$new(rank = k, x_max = 100, learning_rate = 0.1)
  res = model$fit_transform(tcm, n_iter = 3)
  # basic dimensions check
  expect_equal(ncol(res), k)
  expect_equal(nrow(res), nrow(tcm))
  expect_equal(rownames(tcm), rownames(tcm))
  expect_equal(colnames(tcm), colnames(model$components))
})
