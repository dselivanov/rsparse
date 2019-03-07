context("soft_svd/soft_impute")

k = 10
seq_k = seq_len(k)

m = movielens100k[1:100, 1:200]

svd_ground_true = svd(m)
svd_soft_svd = soft_svd(m, rank = k, n_iter = 100, convergence_tol = 1e-6)

test_that("test matrix restore with soft-svd", {
  m_restored_svd = svd_ground_true$u[, seq_k] %*% diag(x = svd_ground_true$d[seq_k]) %*% t(svd_ground_true$v[, seq_k])
  m_restored_soft_svd = svd_soft_svd$u %*% diag(x = svd_soft_svd$d) %*% t(svd_soft_svd$v)

  expect_equal(m_restored_svd, m_restored_soft_svd, tolerance = 1e-1)
})

test_that("test soft-impute", {
  res_soft_impute = soft_impute(movielens100k, rank = k, n_iter = 100, convergence_tol = 1e-3)

  # basic dimesnions check
  expect_equal(length(res_soft_impute$d), k)
  expect_equal(ncol(res_soft_impute$v), k)
  expect_equal(ncol(res_soft_impute$u), k)
  expect_equal(nrow(res_soft_impute$u), nrow(movielens100k))
  expect_equal(nrow(res_soft_impute$v), ncol(movielens100k))

  # check vectors are orthogonal
  expect_equal(sum((crossprod(res_soft_impute$v) - diag(k))**2), 0, tolerance = 1e-6)
  expect_equal(sum((crossprod(res_soft_impute$u) - diag(k))**2), 0, tolerance = 1e-6)
})

