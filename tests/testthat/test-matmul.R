context("matmul")

k = 10
nc = ncol(movielens100k)
nr = nrow(movielens100k)
x_nc = matrix(rep(1:10, nc), nrow = nc)
x_nr = t(matrix(rep(1:10, nr), nrow = nr))

csc = movielens100k
csr = as(movielens100k, "RsparseMatrix")
dense = as.matrix(movielens100k)

test_that("matmul CSR", {
  expect_equal(csr %*% x_nc, dense %*% x_nc)
  expect_equal(tcrossprod(csr, t(x_nc)), tcrossprod(dense, t(x_nc)))
  expect_error(csr %*% cbind(x_nc, rep(1, ncrow(x_nc))))
})

test_that("matmul CSC", {
  expect_equal(x_nr %*% csc, x_nr %*% dense)
  expect_equal(crossprod(t(x_nr), csc), crossprod(t(x_nr), csc))
  expect_error(cbind(x_nr, rep(1, nrow(x_nr))) %*% csc)
})
