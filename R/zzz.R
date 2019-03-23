#' @import methods
#' @import Matrix
#' @importFrom stats rnorm
#' @import data.table
#' @import Rcpp
#' @import float
#' @importFrom RhpcBLASctl get_num_cores
#' @importFrom mlapi mlapiDecomposition
#' @useDynLib rsparse


.onAttach = function(libname, pkgname) {
  n_omp_threads = detect_number_omp_threads()
  if(interactive())
    packageStartupMessage(sprintf("Setting OpenMP threads number to %d. \nCan be adjusted by setting `options(\"rsparse_omp_threads\" = N_THREADS)`", n_omp_threads))

  options("rsparse_omp_threads" = n_omp_threads)

}

.onLoad = function(libname, pkgname) {
  options("rsparse_omp_threads" = detect_number_omp_threads())

  logger = lgr::get_logger('rsparse')
  logger$set_threshold('info')
  assign('logger', logger, envir = parent.env(environment()))
}

#' Detects number of OpenMP threads in the system
#'
#' Detects number of OpenMP threads in the system respecting environment
#' variables such as \code{OMP_NUM_THREADS} and \code{OMP_THREAD_LIMIT}
#'
#' @export
detect_number_omp_threads = function() {
  n_omp_threads = as.numeric(Sys.getenv("OMP_NUM_THREADS"))
  if (is.na(n_omp_threads) || n_omp_threads <= 0) n_omp_threads = omp_thread_count()
  ## if OMP_THREAD_LIMIT is set, maximize on that limit.
  omp_thread_limit = as.numeric(Sys.getenv("OMP_THREAD_LIMIT"))
  if ( is.na(omp_thread_limit) ) omp_thread_limit = n_omp_threads
  n_omp_threads = min(omp_thread_limit, n_omp_threads)

  n_omp_threads
}
