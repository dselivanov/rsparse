Sys.setenv("R_TESTS" = "")
library(testthat)
library(rsparse)
library(data.table)
library(Matrix)
options(rsparse_omp_threads = 1L)
logger = lgr::get_logger('rsparse')
logger$set_threshold('warn')
data("movielens100k")
test_check("rsparse")

movielens100k = rbind(
  movielens100k,
  as(matrix(0, nrow = 2, ncol = ncol(movielens100k)), 'sparseMatrix'),
  movielens100k[1:2, ]
)
