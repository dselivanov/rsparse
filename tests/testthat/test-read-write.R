library("testthat")
library("rsparse")
context("Read and Write matrices")

test_that("Regression mode", {
  txt_mat = paste(
    "-1.234 1:10 4:4.500000000",
    "0 ",
    "1e3 1:.001 2:5e-3",
    sep = "\n"
  )
  r = read.sparse(txt_mat, from_string=TRUE)
  expect_s4_class(r$X, "dgRMatrix")
  expect_type(r$y, "double")

  expected_X = matrix(c(10, 0, 0, 4.5, 0, 0, 0, 0, 0.001, 0.005, 0, 0),
                      nrow=3, ncol=4, byrow=TRUE)
  expected_y = c(-1.234, 0, 1000)
  compare_vals = function(expected_X, expected_y, X, y) {
    expect_equal(expected_X, unname(as.matrix(X)))
    expect_equal(expected_y, y)
  }

  compare_vals(expected_X, expected_y, r$X, r$y)

  file_name = file.path(tempdir(), "test_sparse_matrix.txt")
  write.sparse(file_name, r$X, r$y, integer_labels=FALSE)
  r = read.sparse(file_name, from_string=FALSE)
  compare_vals(expected_X, expected_y, r$X, r$y)

  s = write.sparse(file_name, r$X, r$y, integer_labels=FALSE, to_string=TRUE)
  r = read.sparse(s, from_string=TRUE)
  compare_vals(expected_X, expected_y, r$X, r$y)
})

test_that("Classification mode", {
  txt_mat = paste(
    "1 1:10 4:4.500000000",
    "0 ",
    "2 1:.001 2:5e-3",
    sep = "\n"
  )
  r = read.sparse(txt_mat, from_string=TRUE, integer_labels=TRUE)
  expect_s4_class(r$X, "dgRMatrix")
  expect_type(r$y, "integer")

  expected_X = matrix(c(10, 0, 0, 4.5, 0, 0, 0, 0, 0.001, 0.005, 0, 0),
                      nrow=3, ncol=4, byrow=TRUE)
  expected_y = c(1L, 0L, 2L)
  compare_vals = function(expected_X, expected_y, X, y) {
    expect_equal(expected_X, unname(as.matrix(X)))
    expect_equal(expected_y, y)
  }

  compare_vals(expected_X, expected_y, r$X, r$y)

  file_name = file.path(tempdir(), "test_sparse_matrix.txt")
  write.sparse(file_name, r$X, r$y, integer_labels=TRUE)
  r = read.sparse(file_name, from_string=FALSE)
  compare_vals(expected_X, expected_y, r$X, r$y)

  s = write.sparse(file_name, r$X, r$y, integer_labels=TRUE, to_string=TRUE)
  r = read.sparse(s, from_string=TRUE, integer_labels=TRUE)
  compare_vals(expected_X, expected_y, r$X, r$y)
})

test_that("Multilabel mode", {
  txt_mat = paste(
    "1,2 1:10 4:4.500000000",
    " ",
    "3 1:.001 2:5e-3",
    sep = "\n"
  )
  r = read.sparse(txt_mat, from_string=TRUE, multilabel=TRUE)
  expect_s4_class(r$X, "dgRMatrix")
  expect_s4_class(r$y, "ngRMatrix")

  expected_X = matrix(c(10, 0, 0, 4.5, 0, 0, 0, 0, 0.001, 0.005, 0, 0),
                      nrow=3, ncol=4, byrow=TRUE)
  expected_y = matrix(c(1,1,0,0,0,0,0,0,1),
                      nrow=3, ncol=3, byrow=TRUE)
  compare_vals = function(expected_X, expected_y, X, y) {
    y = unname(as.matrix(y))
    mode(y) = "double"
    expect_equal(expected_X, unname(as.matrix(X)))
    expect_equal(expected_y, y)
  }

  compare_vals(expected_X, expected_y, r$X, r$y)

  file_name = file.path(tempdir(), "test_sparse_matrix.txt")
  write.sparse(file_name, r$X, r$y)
  r = read.sparse(file_name, from_string=FALSE, multilabel=TRUE)
  compare_vals(expected_X, expected_y, r$X, r$y)

  s = write.sparse(file_name, r$X, r$y, to_string=TRUE)
  r = read.sparse(s, from_string=TRUE, multilabel=TRUE)
  compare_vals(expected_X, expected_y, r$X, r$y)
})

test_that("Ranking mode", {
  txt_mat = paste(
    "1 qid:1 1:10 4:4.500000000",
    "0 qid:2",
    "2 qid:1 1:.001 2:5e-3",
    sep = "\n"
  )
  r = read.sparse(txt_mat, from_string=TRUE, integer_labels=TRUE, has_qid=TRUE)
  expect_s4_class(r$X, "dgRMatrix")
  expect_type(r$y, "integer")
  expect_type(r$qid, "integer")

  expected_X = matrix(c(10, 0, 0, 4.5, 0, 0, 0, 0, 0.001, 0.005, 0, 0),
                      nrow=3, ncol=4, byrow=TRUE)
  expected_y = c(1L, 0L, 2L)
  expected_qid = c(1L, 2L, 1L)
  compare_vals = function(expected_X, expected_y, expected_qid, X, y, qid) {
    expect_equal(expected_X, unname(as.matrix(X)))
    expect_equal(expected_y, y)
    expect_equal(expected_qid, qid)
  }

  compare_vals(expected_X, expected_y, expected_qid, r$X, r$y, r$qid)

  file_name = file.path(tempdir(), "test_sparse_matrix.txt")
  write.sparse(file_name, r$X, r$y, r$qid, integer_labels=TRUE)
  r = read.sparse(file_name, from_string=FALSE, has_qid=TRUE)
  compare_vals(expected_X, expected_y, expected_qid, r$X, r$y, r$qid)

  s = write.sparse(file_name, r$X, r$y, r$qid, integer_labels=TRUE, to_string=TRUE)
  r = read.sparse(s, from_string=TRUE, integer_labels=TRUE, has_qid=TRUE)
  compare_vals(expected_X, expected_y, expected_qid, r$X, r$y, r$qid)
})
