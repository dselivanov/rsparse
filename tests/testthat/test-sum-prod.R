library("testthat")
library("Matrix")
library("rsparse")
context("Adding and multiplying matrices")

test_that("Operations matrix-matrix", {
  set.seed(1)
  Mat1 = rsparsematrix(100, 35, .4)
  Mat2 = rsparsematrix(100, 35, .6)
  csr1 = as.csr.matrix(Mat1)
  csc1 = as.csc.matrix(Mat1)
  mat1 = as.matrix(Mat1)
  csr2 = as.csr.matrix(Mat2)
  csc2 = as.csc.matrix(Mat2)
  mat2 = as.matrix(Mat2)
  emat = new("dgRMatrix")
  emat@Dim = csr1@Dim
  emat@Dimnames = csr1@Dimnames
  emat@p = integer(length(csr1@p))
  eden = as.matrix(emat)

  expect_equal(as.matrix(csr1 + csr2), as.matrix(csc1 + csc2))
  expect_equal(as.matrix(csr1 + csc2), as.matrix(csc1 + csc2))
  expect_equal(as.matrix(csr1 + csr1), as.matrix(csc1 + csr1))
  expect_equal(as.matrix(csr1 + emat), as.matrix(csc1 + emat))
  expect_equal(as.matrix(emat + emat), as.matrix(eden + eden))
  expect_s4_class(csr1 + csr2, "dgRMatrix")
  expect_s4_class(csr1 + csc2, "dgRMatrix")
  expect_s4_class(csr1 + csr1, "dgRMatrix")
  expect_s4_class(csr1 + emat, "dgRMatrix")
  expect_s4_class(emat + emat, "dgRMatrix")

  expect_equal(as.matrix(csr1 - csr2), as.matrix(csc1 - csc2))
  expect_equal(as.matrix(csr1 - csc2), as.matrix(csc1 - csc2))
  expect_equal(as.matrix(csr1 - csr1), as.matrix(csc1 - csr1))
  expect_equal(as.matrix(csr1 - emat), as.matrix(csc1 - emat))
  expect_equal(as.matrix(emat - emat), as.matrix(eden - eden))
  expect_s4_class(csr1 - csr2, "dgRMatrix")
  expect_s4_class(csr1 - csc2, "dgRMatrix")
  expect_s4_class(csr1 - csr1, "dgRMatrix")
  expect_s4_class(csr1 - emat, "dgRMatrix")
  expect_s4_class(emat - emat, "dgRMatrix")

  expect_equal(as.matrix(csr1 * csr2), as.matrix(csc1 * csc2))
  expect_equal(as.matrix(csr1 * csc2), as.matrix(csc1 * csc2))
  expect_equal(as.matrix(csr1 * csr1), as.matrix(csc1 * csr1))
  expect_equal(as.matrix(csr1 * emat), as.matrix(csc1 * emat))
  expect_equal(as.matrix(emat * emat), as.matrix(eden * eden))
  expect_s4_class(csr1 * csr2, "dgRMatrix")
  expect_s4_class(csr1 * csc2, "dgRMatrix")
  expect_s4_class(csr1 * csr1, "dgRMatrix")
  expect_s4_class(csr1 * emat, "dgRMatrix")
  expect_s4_class(emat * emat, "dgRMatrix")

  expect_equal(as.matrix(csr1^2), as.matrix(csc1^2))
  expect_equal(as.matrix(sqrt(csr1^2)), as.matrix(sqrt(csc1^2)))
  expect_equal(as.matrix(abs(csr1)), as.matrix(abs(csc1)))
  expect_s4_class(csr1^2, "dgRMatrix")
  expect_s4_class(sqrt(csr1^2), "dgRMatrix")
  expect_s4_class(abs(csr1), "dgRMatrix")

  expect_equal(as.matrix(emat^2), as.matrix(eden^2))
  expect_equal(as.matrix(sqrt(emat^2)), as.matrix(sqrt(eden^2)))
  expect_equal(as.matrix(abs(emat)), as.matrix(abs(eden)))
  expect_s4_class(emat^2, "dgRMatrix")
  expect_s4_class(sqrt(emat^2), "dgRMatrix")
  expect_s4_class(abs(emat), "dgRMatrix")
})

test_that("Types of objects", {
  set.seed(1)
  Mat = rsparsematrix(5, 3, .4)
  Mat = as.csr.matrix(Mat)

  expect_s4_class(Mat + as.csr.matrix(Mat, binary=TRUE), "dgRMatrix")
  expect_equal(Mat + as.csr.matrix(Mat, binary=TRUE),
               Mat + as.csr.matrix(as.csr.matrix(Mat, binary=TRUE)))
  expect_s4_class(Mat - as.csr.matrix(Mat, binary=TRUE), "dgRMatrix")
  expect_equal(Mat - as.csr.matrix(Mat, binary=TRUE),
               Mat - as.csr.matrix(as.csr.matrix(Mat, binary=TRUE)))
  expect_s4_class(Mat * as.csr.matrix(Mat, binary=TRUE), "dgRMatrix")
  expect_equal(Mat * as.csr.matrix(Mat, binary=TRUE),
               Mat * as.csr.matrix(as.csr.matrix(Mat, binary=TRUE)))

  expect_s4_class(Mat + as.csr.matrix(Mat, logical=TRUE), "dgRMatrix")
  expect_equal(Mat + as.csr.matrix(Mat, logical=TRUE),
               Mat + as.csr.matrix(as.csr.matrix(Mat, logical=TRUE)))
  expect_s4_class(Mat - as.csr.matrix(Mat, logical=TRUE), "dgRMatrix")
  expect_equal(Mat - as.csr.matrix(Mat, logical=TRUE),
               Mat - as.csr.matrix(as.csr.matrix(Mat, logical=TRUE)))
  expect_s4_class(Mat * as.csr.matrix(Mat, logical=TRUE), "dgRMatrix")
  expect_equal(Mat * as.csr.matrix(Mat, logical=TRUE),
               Mat * as.csr.matrix(as.csr.matrix(Mat, logical=TRUE)))




  expect_s4_class(as.csr.matrix(Mat, binary=TRUE) + as.csr.matrix(Mat, logical=TRUE), "dgRMatrix")
  expect_equal(as.csr.matrix(Mat, binary=TRUE) + as.csr.matrix(Mat, logical=TRUE),
               as.csr.matrix(Mat, binary=TRUE) + as.csr.matrix(as.csr.matrix(Mat, logical=TRUE)))
  expect_s4_class(as.csr.matrix(Mat, binary=TRUE) - as.csr.matrix(Mat, logical=TRUE), "dgRMatrix")
  expect_equal(as.csr.matrix(Mat, binary=TRUE) - as.csr.matrix(Mat, logical=TRUE),
               as.csr.matrix(Mat, binary=TRUE) - as.csr.matrix(as.csr.matrix(Mat, logical=TRUE)))
  expect_s4_class(as.csr.matrix(Mat, binary=TRUE) * as.csr.matrix(Mat, logical=TRUE), "dgRMatrix")
  expect_equal(as.csr.matrix(Mat, binary=TRUE) * as.csr.matrix(Mat, logical=TRUE),
               as.csr.matrix(Mat, binary=TRUE) * as.csr.matrix(as.csr.matrix(Mat, logical=TRUE)))

})
