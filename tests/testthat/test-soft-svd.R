context("soft_svd")

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

