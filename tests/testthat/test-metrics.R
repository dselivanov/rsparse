context("ranking metrics")

predictions = matrix(
  c(5L, 7L, 9L, 2L),
  nrow = 1
)

test_that("test ap_k", {
  actual = matrix(
    c(0, 0, 0, 0, 1, 0, 1, 0, 1, 0),
    nrow = 1
  )
  actual = as(actual, "RsparseMatrix")
  expect_equal(rsparse::ap_k(predictions, actual), 1)

  actual_2 = actual
  actual_2[1, 10] = 1
  ap_k_2 = rsparse::ap_k(predictions, actual_2)
  expect_lt(ap_k_2, 1)

  actual_3 = actual
  actual_3[1, 1] = 1
  ap_k_3 = rsparse::ap_k(predictions, actual_3)
  expect_equal(ap_k_2, ap_k_3)
})

test_that("test ndcg_k", {
  actual = matrix(
    c(0, 0, 0, 0, 10, 0, 8, 0, 4, 0),
    nrow = 1
  )
  actual = as(actual, "RsparseMatrix")
  expect_equal(rsparse::ndcg_k(predictions, actual), 1)

  actual_2 = actual
  actual_2[1, 5] = 1
  ndcg_k_2 = rsparse::ndcg_k(predictions, actual_2)
  expect_lt(ndcg_k_2, 1)

  actual_3 = actual
  actual_3[1, 7] = 1
  ndcg_k_3 = rsparse::ndcg_k(predictions, actual_3)
  expect_gt(ndcg_k_3, ndcg_k_2)
})
