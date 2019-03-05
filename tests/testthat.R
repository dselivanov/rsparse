Sys.setenv("R_TESTS" = "")
library(testthat)
library(rsparse)
library(data.table)
library(Matrix)
data("movielens100k")
test_check("rsparse")

futile.logger::flog.threshold(futile.logger::WARN)
