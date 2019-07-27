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
  install_name_tool_change_float()
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

# see https://github.com/dselivanov/rsparse/issues/25
install_name_tool_change_float = function() {
  if (Sys.info()[["sysname"]] == "Darwin") {
    install_name_tool = 'install_name_tool'
    rsparse_path = system.file('libs', package = 'rsparse')
    rsparse_path = file.path(rsparse_path, 'rsparse.so')
    # find reference to float.so
    otool_res = system2('otool', sprintf('-l %s', rsparse_path), stdout = TRUE)
    i = grepl('float.so', otool_res, fixed = T)
    float_rpath = otool_res[i]
    # extract '@rpath' part out of string like (binary installation)
    # 'name @rpath/Volumes/SSD-Data/Builds/R-dev-web/QA/Simon/packages/el-capitan-x86_64/Rlib/3.6/float/libs/float.so (offset 24)'
    # remove everything before '@rpath'
    float_rpath = gsub(".*name ", "", float_rpath)
    # remove everything inside brackets and space at the end
    float_rpath = gsub("\\([^()]*\\)|\\s", "", float_rpath)
    new_float_rpath = system.file('libs', package = 'float')
    new_float_rpath = file.path(new_float_rpath, 'float.so')
    # useful info - https://github.com/conda/conda-build/issues/279#issuecomment-67241554
    install_name_tool_args = sprintf('-change %s %s %s', float_rpath, new_float_rpath, rsparse_path)
    invisible(system2(install_name_tool, install_name_tool_args))
  }
}
