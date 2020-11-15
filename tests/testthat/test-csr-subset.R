library("testthat")
library("rsparse")
context("RsparseMatrix subsets")

nc = 500L
nr = 1000L
set.seed(123)
m = Matrix::rsparsematrix(nrow = nr, ncol = nc, density = 0.1)
colnames(m) = as.character(seq_len(nc))
rownames(m) = as.character(seq_len(nr))
m = as(m, "RsparseMatrix")
m_csc = as(m, "CsparseMatrix")
m_base = as.matrix(m)

test_that("RsparseMatrix subset cols and rows", {
  expect_equal(m, m[, ])
  expect_equal(m, m[])
  expect_equal(m, m[, , ])
  expect_equal(m[1:10, 1:100], as(m_csc[1:10, 1:100], "RsparseMatrix"))
  expect_equal(m[as.character(1:10), 1:100], as(m_csc[as.character(1:10), 1:100], "RsparseMatrix"))
  expect_equal(m["10", "20", drop = FALSE], as(m_csc["10", "20", drop = FALSE], "RsparseMatrix"))
  expect_equal(m["10", "20", drop = TRUE], m_csc["10", "20", drop = TRUE])
})

test_that("RsparseMatrix subset non sequential", {
  expect_equal(m, m[, ])
  expect_equal(m, m[])
  expect_equal(m, m[, , ])
  expect_equal(m[c(5,2,1,7,4), c(5,2,1,7,4,10,100)],
               as(m_base[c(5,2,1,7,4), c(5,2,1,7,4,10,100)], "RsparseMatrix"))
  expect_equal(m[as.character(c(5,2,1,7,4)), as.character(c(5,2,1,7,4,10,100))],
               as(m_base[c(5,2,1,7,4), c(5,2,1,7,4,10,100)], "RsparseMatrix"))
})

test_that("RsparseMatrix subset repeated", {
  expect_equal(m, m[, ])
  expect_equal(m, m[])
  expect_equal(m, m[, , ])
  expect_equal(m[c(2,2,2,1,1,3), c(3,3,4,4,1,1,1)],
               as(m_base[c(2,2,2,1,1,3), c(3,3,4,4,1,1,1)], "RsparseMatrix"))
  expect_equal(m[as.character(c(2,2,2,1,1,3)), as.character(c(3,3,4,4,1,1,1))],
               as(m_base[c(2,2,2,1,1,3), c(3,3,4,4,1,1,1)], "RsparseMatrix"))
  expect_equal(m[c(5,2,1,7,4,1,5), c(5,2,1,7,4,1,10,100,5)],
               as(m_base[c(5,2,1,7,4,1,5), c(5,2,1,7,4,1,10,100,5)], "RsparseMatrix"))
  expect_equal(m[as.character(c(5,2,1,7,4,1,5)), as.character( c(5,2,1,7,4,1,10,100,5))],
               as(m_base[c(5,2,1,7,4,1,5),  c(5,2,1,7,4,1,10,100,5)], "RsparseMatrix"))
})

test_that("RsparseMatrix subset empty", {
  expect_equal(m[3:10, integer()],
               as(m_base[3:10, integer()], "RsparseMatrix"))
  expect_equal(m[c(2,2,2,1,1,3), integer()],
               as(m_base[c(2,2,2,1,1,3), integer()], "RsparseMatrix"))
  expect_equal(m[, integer()],
               as(m_base[, integer()], "RsparseMatrix"))

  expect_equal(m[character(), ],
               as(m_base[integer(), ], "RsparseMatrix"))
  expect_equal(m[character(), as.character(c(3,3,4,4,1,1,1))],
               as(m_base[integer(), c(3,3,4,4,1,1,1)], "RsparseMatrix"))
  expect_equal(m[character(), 3:10],
               as(m_base[integer(), 3:10], "RsparseMatrix"))

  expect_equal(m[integer(), integer()],
               as(m_base[integer(), integer()], "RsparseMatrix"))
  expect_equal(m[character(), character()],
               as(m_base[character(),  character()], "RsparseMatrix"))
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
