# What is this?

R package wich aims to implement several algrithms for recommender systems. 

Where possible parallelization is also applied (via OpenMP and multithreaded BLAS).

# Tutorials

1. [Introduction to matrix factorization with Weighted-ALS algorithm](http://dsnotes.com/post/2017-05-28-matrix-factorization-for-recommender-systems/) - collaborative filtering for implicit feedback datasets.
1. [Music recommendations using LastFM-360K dataset](http://dsnotes.com/post/2017-06-28-matrix-factorization-for-recommender-systems-part-2/)
    * evaluation metrics for ranking
    * setting up proper cross-validation
    * possible issues with nested parallelism and thread contention
    * making recommendations for new users
    * complimentary item-to-item recommendations

# Algorithms

At the moment following algorithms are implemented:

### Alternating Least Squares for implicit feedback

Current implementation used RcppArmadillo and  **extensively uses BLAS and LAPACK**, so on my 4-core PC with OpenBLAS it is **~1.7x faster** than highly optimized Quora's [qmf](https://github.com/quora/qmf) library.

See [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) paper by (Yifan Hu, Yehuda Koren, Chris Volinsky) for details.  

**VERY IMPORTANT** if you use multithreaded BLAS (you generally should) such as OpenBLAS, Intel MKL, Apple Accelerate, I **highly recommend disable its internal multithreading ability**. This leads to **substantial speedups** (can be 10x!) for this package (since matrix factorization is already parallelized in package with OpenMP). This can be done by setting corresponding environment variables **before starting `R`**:

1. OpenBLAS: `export OPENBLAS_NUM_THREADS=1`.
1. Intel MKL: `export MKL_NUM_THREADS=1`
1. Apple Accelerate: `export VECLIB_MAXIMUM_THREADS=1`

It it also possible to change number of threads in runtime, see for example [OpenBlasThreads](https://github.com/rundel/OpenBlasThreads) and [RhpcBLASctl](https://cran.r-project.org/web/packages/RhpcBLASctl/index.html) packages.

### Alternating Least Squares for explicit feedback

# API

We follow [mlapi](https://github.com/dselivanov/mlapi) conventions.

# Quice reference

TODO
