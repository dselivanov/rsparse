#' @import methods
#' @importFrom stats rnorm
#' @import data.table
#' @import Rcpp
#' @import float
#' @import Matrix
#' @importFrom RhpcBLASctl get_num_cores
#' @useDynLib rsparse, .registration = TRUE


.onAttach = function(libname, pkgname) {
  n_omp_threads = getOption("rsparse_omp_threads")
  if (is.null(n_omp_threads)) { # unlikely given .onLoad below
    n_omp_threads = detect_number_omp_threads()
    options("rsparse_omp_threads" = n_omp_threads)
  }
  if (interactive()) {
    msg = paste0("Number of OpenMP threads set to ",
                 n_omp_threads,
                 "\nCan be adjusted by setting `options(\"rsparse_omp_threads\" = N_THREADS)`")
    packageStartupMessage(msg)
  }
}

.onLoad = function(libname, pkgname) {
  # install_name_tool_change_float()
  # library.dynam("rsparse", pkgname, libname)
  if (is.null(getOption("rsparse_omp_threads")))
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
  # now respects both OMP_NUM_THREADS and OMP_THREAD_LIMIT
  omp_thread_count()
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
