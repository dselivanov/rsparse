# New submission

- fixed HTML validation problems discovered by CRAN checks
- fixed NEWS.md to follow CRAN format
- failing tests on "r-oldrel-windows-ix86+x86_64" are due to the lack of `MatrixExtra` library. I'm not sure why this happened on CRAN server

# Test environments

- laptop OS X, R 4.0.5
- win-builder (devel)

# R CMD check results

One misc note which shows 403 error when accessing [http://yifanhu.net/PUB/cf.pdf](http://yifanhu.net/PUB/cf.pdf). This seems CRAN server specific.
