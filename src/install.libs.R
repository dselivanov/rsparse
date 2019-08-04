### Modified from pbdZMQ/src/install.libs.R
### For libs
files <- c("rsparse.so", "rsparse.so.dSYM", "rsparse.dylib", "rsparse.dll",
           "symbols.rds")
files <- files[file.exists(files)]

if(length(files) > 0){
  libsarch <- if (nzchar(R_ARCH)) paste("libs", R_ARCH, sep='') else "libs"
  dest <- file.path(R_PACKAGE_DIR, libsarch)
  dir.create(dest, recursive = TRUE, showWarnings = FALSE)
  file.copy(files, dest, overwrite = TRUE, recursive = TRUE)

  ### Overwrite RPATH from the shared library installed to the destination.
  if(Sys.info()[['sysname']] == "Darwin"){
    cmd.int <- system("which install_name_tool", intern = TRUE)
    cmd.ot <- system("which otool", intern = TRUE) 
    fn.rsparse.so <- file.path(dest, "rsparse.so")

    if(file.exists(fn.rsparse.so)){
      dest.float <- file.path(system.file(package = "float"), libsarch)
      fn.float.so <- file.path(dest.float, "float.so") 
      if(! file.exists(fn.float.so)){
        stop(paste(fn.float.so, " does not exist.", sep = ""))
      }

      ### For "rsparse.so"
      rpath <- system(paste(cmd.ot, " -L ", fn.rsparse.so, sep = ""),
                      intern = TRUE)
      cat("\nBefore install_name_tool (install.libs.R & rsparse.so):\n")
      print(rpath)

      cmd <- paste(cmd.int, " -change float.so ", fn.float.so, " ",
                   fn.rsparse.so, sep = "")
      cat("\nIn install_name_tool (install.libs.R & rsparse.so & path):\n")
      print(cmd) 
      system(cmd)

      rpath <- system(paste(cmd.ot, " -L ", fn.rsparse.so, sep = ""),
                      intern = TRUE)
      cat("\nAfter install_name_tool (install.libs.R & rsparse.so):\n")
      print(rpath)
    }
  }
}

