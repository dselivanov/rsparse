New submission

- fixed illegal memory access as reported by CRAN UBSAN checks

# Test environments

- fedora distribution from rhub
- debian based docker with r-devel based (from rocker)
- local ubuntu 16.04, R 3.5.1
- laptop OS X, R 3.4.0
- win-builder (devel)
- R-devel ubsan-clang (https://hub.docker.com/r/rocker/r-devel-ubsan-clang/)

# R CMD check results

0 errors | 0 warning | 1 note

> checking installed package size ... NOTE
    installed size is  8.2Mb
    sub-directories of 1Mb or more:
      libs   7.5Mb

Not sure I can do something about this - C++ pkg with Armadillo and float dependencies
