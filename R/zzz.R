#' @import methods
#' @import Matrix
#' @importFrom stats rnorm
#' @import data.table
#' @import Rcpp
#' @import float
#' @importFrom RhpcBLASctl get_num_cores
#' @useDynLib rsparse, .registration = TRUE


.onAttach = function(libname, pkgname) {
  n_omp_threads = detect_number_omp_threads()
  if (interactive()) {
    msg = paste0("Setting OpenMP threads number to ",
                 n_omp_threads,
                 "\nCan be adjusted by setting `options(\"rsparse_omp_threads\" = N_THREADS)`")
    packageStartupMessage(msg)
  }

  options("rsparse_omp_threads" = n_omp_threads)

}

.onLoad = function(libname, pkgname) {
  # install_name_tool_change_float()
  # library.dynam("rsparse", pkgname, libname)
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

install_name_tool_change_float = function() {
  if (Sys.info()[["sysname"]] == "Darwin") {
    R_ARCH = Sys.getenv("R_ARCH")
    libsarch = if (nzchar(R_ARCH)) paste("libs", R_ARCH, sep = "") else "libs"

    dest.rsparse = file.path(system.file(package = "rsparse"), libsarch)
    fn.rsparse.so = file.path(dest.rsparse, "rsparse.so")

    dest.float = file.path(system.file(package = "float"), libsarch)
    fn.float.so = file.path(dest.float, "float.so")

    message("fn.float.so ", fn.float.so)

    cmd.int = system("which install_name_tool", intern = TRUE)
    cmd.ot = system("which otool", intern = TRUE)

    rpath = system(paste(cmd.ot, " -L ", fn.rsparse.so, sep = ""), intern = TRUE)
    message("rpath ", rpath)

    id = grep("float.so", rpath)
    fn.float.so.org = gsub("^\\t(.*float.so) \\(.*\\)$", "\\1", rpath[id])

    message("fn.float.so.org ", fn.float.so.org)

    ret = NULL
    if (fn.float.so.org != fn.float.so) {
      print(fn.float.so.org)
      print(fn.float.so)
      cmd = paste(cmd.int, " -change ", fn.float.so.org, " ", fn.float.so, " ",
                   fn.rsparse.so, sep = "")
      ret = system(cmd, intern = TRUE)
    }
    invisible(ret)
  }
}
