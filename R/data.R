#' MovieLens 100K Dataset
#'
#' This data set consists of:
#' \enumerate{
#'   \item 100,000 ratings (1-5) from 943 users on 1682 movies.
#'   \item Each user has rated at least 20 movies.
#' }
#' MovieLens data sets were collected by the GroupLens Research Project at the University of Minnesota.
#'
#' @name movielens100k
#' @usage data("movielens100k")
#' @format A sparse column-compressed matrix (\code{Matrix::dgCMatrix}) with 943 rows and 1682 columns.
#' \enumerate{
#'   \item rows are users
#'   \item columns are movies
#'   \item values are ratings
#' }
#' @source \url{https://grouplens.org/datasets/movielens/100k/}
#' @keywords datasets
NULL
