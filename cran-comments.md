New submission

- another way of linking to float shared library on binary OS X installations

# Test environments

- local ubuntu 16.04, R 3.6.0
- laptop OS X, R 3.6.0
- win-builder (devel)

# R CMD check results

0 errors | 0 warning | 1 note

> checking installed package size ... NOTE
    installed size is  8.2Mb
    sub-directories of 1Mb or more:
      libs   7.5Mb

Not sure I can do something about this - C++ pkg with Armadillo and float dependencies
