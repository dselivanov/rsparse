library("testthat")
library("rsparse")
context("RsparseMatrix subsets")

nc = 500L
nr = 1000L
m = Matrix::rsparsematrix(nrow = nr, ncol = nc, density = 0.01)
colnames(m) = as.character(seq_len(nc))
rownames(m) = as.character(seq_len(nr))
m = as(m, "RsparseMatrix")
m_csc = as(m, "CsparseMatrix")

test_that("RsparseMatrix subset cols and rows", {
  expect_equal(m, m[, ])
  expect_equal(m, m[])
  expect_equal(m, m[, , ])
  expect_equal(m[1:10, 1:100], as(m_csc[1:10, 1:100], "RsparseMatrix"))
  expect_equal(m[as.character(1:10), 1:100], as(m_csc[as.character(1:10), 1:100], "RsparseMatrix"))
  expect_equal(m["10", "20", drop = FALSE], as(m_csc["10", "20", drop = FALSE], "RsparseMatrix"))
  expect_equal(m["10", "20", drop = TRUE], m_csc["10", "20", drop = TRUE])
})

test_that("RsparseMatrix subset cols", {
  expect_true(inherits(m[, 2L], 'numeric'))
  expect_true(inherits(m[, 2L, drop = FALSE], 'RsparseMatrix'))
  expect_true(inherits(m[, 1L:2L], 'RsparseMatrix'))
  expect_equal(rownames(m[, 2L:4L]), rownames(m))
  expect_equal(colnames(m[, 2L:4L]), as.character(2L:4L) )
  expect_equal(m[, as.character(2L:4L)], m[, 2L:4L])
  expect_error(m[, 501L])
  expect_error(m[, 500L:501L])
  expect_equal(m[, -1, drop = FALSE], m[, ncol(m), drop = FALSE])
  expect_equal(m[, -1, drop = TRUE], m[, ncol(m), drop = TRUE])
  expect_equal(m[, -10:-1 ], m[, (ncol(m) - 9):ncol(m)])
})

test_that("RsparseMatrix subset rows", {
  expect_true(inherits(m[2L, ], 'numeric'))
  expect_true(inherits(m[2L, , drop = FALSE], 'RsparseMatrix'))
  expect_true(inherits(m[1L:2L, ], 'RsparseMatrix'))
  expect_equal(colnames(m[2L:4L, ]), colnames(m))
  expect_equal(rownames(m[2L:4L, ]), as.character(2L:4L) )
  expect_equal(m[as.character(2L:4L), ], m[2L:4L, ] )
  expect_error(m[1001L, ])
  expect_error(m[900L:1001L, ])
  expect_equal(m[-1, , drop = TRUE], m[nrow(m), , drop = TRUE])
  expect_equal(m[-1, , drop = TRUE], m[nrow(m), , drop = TRUE])
  expect_equal(m[-10:-1, ], m[(nrow(m) - 9):nrow(m), ])
})
