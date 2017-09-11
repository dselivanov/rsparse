# reco

`reco` is an R package which implements several algrithms for matrix factorization targeting recommender systems. 

1. Weighted Regularized Matrix Factorization (WRMF) from [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) (by Yifan Hu, Yehuda Koren, Chris Volinsky). One of the most efficient (benchmarks below) solvers.
1. Linear-Flow from [Practical Linear Models for Large-Scale One-Class Collaborative Filtering](http://www.bkveton.com/docs/ijcai2016.pdf). This algorithm is similar to [SLIM](http://glaros.dtc.umn.edu/gkhome/node/774) but looks for factorized low-rank item-item similarity matrix.
1. Regularized Matrix Factorization (MF) - classic approch for "rating" prediction.

Package is **quite fast**:

* Built on top of `RcppArmadillo`
* extensively use **BLAS** and parallelized with **OpenMP**
* implements **Conjugate Gradient solver** as dicribed in [Applications of the Conjugate Gradient Method for Implicit
Feedback Collaborative Filtering](https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf) and [Faster Implicit Matrix Factorization](www.benfrederickson.com/fast-implicit-matrix-factorization/)
* Top-k items inference is `O(n*log(k))` and use **BLAS** + **OpenMP**

![benchmark](https://github.com/dselivanov/bench-wals/raw/master/img/wals-bench-cg.png)

# Tutorials

1. [Introduction to matrix factorization with Weighted-ALS algorithm](http://dsnotes.com/post/2017-05-28-matrix-factorization-for-recommender-systems/) - collaborative filtering for implicit feedback datasets.
1. [Music recommendations using LastFM-360K dataset](http://dsnotes.com/post/2017-06-28-matrix-factorization-for-recommender-systems-part-2/)
    * evaluation metrics for ranking
    * setting up proper cross-validation
    * possible issues with nested parallelism and thread contention
    * making recommendations for new users
    * complimentary item-to-item recommendations
1. [Benchmark](http://dsnotes.com/post/2017-07-10-bench-wrmf/) against other good implementations


# API

We follow [mlapi](https://github.com/dselivanov/mlapi) conventions.

# Notes on multithreading and BLAS

**VERY IMPORTANT** if you use multithreaded BLAS (you generally should) such as OpenBLAS, Intel MKL, Apple Accelerate, I **highly recommend disable its internal multithreading ability**. This leads to **substantial speedups** for this package (can be easily 10x and more). Matrix factorization is already parallelized in package with OpenMP. This can be done by setting corresponding environment variables **before starting `R`**:

1. OpenBLAS: `export OPENBLAS_NUM_THREADS=1`.
1. Intel MKL: `export MKL_NUM_THREADS=1`
1. Apple Accelerate: `export VECLIB_MAXIMUM_THREADS=1`

It it also possible to change number of threads in runtime, see for example following packages:

* [OpenBlasThreads](https://github.com/rundel/OpenBlasThreads)
* [RhpcBLASctl](https://cran.r-project.org/web/packages/RhpcBLASctl/index.html)
