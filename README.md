# What is this?

R package wich aims to implement several algrithms for recommender systems. 

Our target is to be within 5x of highly optimized C/C++ implementations. Where possible parallelization is also applied (but keep in mind what we don't care too much about parallelization on Windows platform. However contributions are welcome).

# Algorithms

At the moment following algorithms are implemented:

### Alternating Least Squares for implicit feedback

Current implementation is **pure `R`, but extensively uses BLAS and LAPACK**, so on my 4-core PC it is only 2x slower than highly optimized Quora's [qmf](https://github.com/quora/qmf) library.

See [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) paper by (Yifan Hu, Yehuda Koren, Chris Volinsky) for details.  

**VERY IMPORTANT** if you use multithreaded BLAS (you generally should) such as OpenBLAS, Intel MKL, Apple Accelerate, I **highly recommend disable its internal multithreading ability**. This leads to **substantial speedups** for this package (since matrix factorization is already parallelized in package on higher level). This can be done by setting corresponding environment variables **before starting `R`**:

1. OpenBLAS: `export OPENBLAS_NUM_THREADS=1`.
1. Intel MKL: `export MKL_NUM_THREADS=1`
1. Apple Accelerate: `export VECLIB_MAXIMUM_THREADS=1`

It it also possible to change number of threads in runtime, see for example [OpenBlasThreads](https://github.com/rundel/OpenBlasThreads) and [RhpcBLASctl](https://cran.r-project.org/web/packages/RhpcBLASctl/index.html) packages.

# API

We follow [mlapi](https://github.com/dselivanov/mlapi) conventions.

# Quice reference

TODO
