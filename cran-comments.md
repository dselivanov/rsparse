Resubmission

- replaced backticks by single quotes, added period at the of description as suggested by CRAN team

# Test environments

- fedora distribution from rhub
- debian based docker with r-devel based (from rocker)
- local ubuntu 16.04, R 3.5.1
- laptop OS X, R 3.4.0
- win-builder (devel)

# R CMD check results

0 errors | 0 warning | 2 notes

> checking installed package size ... NOTE
    installed size is  8.5Mb
    sub-directories of 1Mb or more:
      libs   7.8Mb

Not sure I can do something about this - C++ pkg with Armadillo and float dependencies
