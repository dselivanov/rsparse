context("top product")

test_that("top product", {
  nr = 100
  k = 10
  nc = 50
  m1 = matrix(runif(nr * k), ncol = k)
  m2 = matrix(runif(nc * k), ncol = nc)

  m3 = m1 %*% m2
  m3_top = rsparse:::top_product(m1, m2, k = k, n_threads = 1, not_recommend_r = new("dgRMatrix"), exclude = integer())

  expect_equal(m3_top[1, ], order(m3[1, ], decreasing = T)[1:k])
})
