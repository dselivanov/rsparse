# reco

`reco` is an R package which implements many algorithms for **sparse matrix factorizations**. Focus is on applications for **recommender systems**.

## Algorithms

1. Vanilla **Maximum Margin Matrix Factorization** - classic approch for "rating" prediction. See `WRMF` class and constructor option `feedback = "explicit"`. Original paper which indroduced MMMF could be found [here](http://ttic.uchicago.edu/~nati/Publications/MMMFnips04.pdf).
    * <img src="docs/img/MMMF.png" width="400">
1. **Weighted Regularized Matrix Factorization (WRMF)** from [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf). See `WRMF` class and constructor option `feedback = "implicit"`. 
We provide 2 solvers:
    1. Exact based of Cholesky Factorization
    1. Approximated based on fixed number of steps of **Conjugate Gradient**.
See details in [Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering](https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf) and [Faster Implicit Matrix Factorization](www.benfrederickson.com/fast-implicit-matrix-factorization/).
    * <img src="docs/img/WRMF.png" width="400">
1. **Linear-Flow** from [Practical Linear Models for Large-Scale One-Class Collaborative Filtering](http://www.bkveton.com/docs/ijcai2016.pdf). Algorithm looks for factorized low-rank item-item similarity matrix (in some sense it is similar to [SLIM](http://glaros.dtc.umn.edu/gkhome/node/774))
    * <img src="docs/img/LinearFlow.png" width="300">
1. **Soft-SVD** via fast Alternating Least Squares as described in [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](https://arxiv.org/pdf/1410.2596.pdf).
    * <img src="docs/img/soft-svd.png" width="600">
1. **Soft-Impute** via fast Alternating Least Squares as described in [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](https://arxiv.org/pdf/1410.2596.pdf).
    * <img src="docs/img/soft-impute.png" width="400">
    * with a solution in SVD form <img src="docs/img/soft-impute-svd-form.png" width="150">


## Efficiency

Package is reasonably fast and scales nicely to datasets with millions of rows and millions of columns:

* built on top of `RcppArmadillo`
* extensively uses **BLAS** and parallelized with **OpenMP**

Here is example of `reco::WRMF` on [lastfm360k](https://www.upf.edu/web/mtg/lastfm360k) dataset in comparison with other good implementations:

<img src="https://github.com/dselivanov/bench-wals/raw/master/img/wals-bench-cg.png" width="600">

# Materials

**Note that syntax could be not up to date since package is under active development**

1. [Slides from DataFest Tbilisi(2017-11-16)](https://www.slideshare.net/DmitriySelivanov/matrix-factorizations-for-recommender-systems)
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

### Notes on multithreading and BLAS

**VERY IMPORTANT** if you use multithreaded BLAS (you generally should) such as OpenBLAS, Intel MKL, Apple Accelerate, I **highly recommend disable its internal multithreading ability**. This leads to **substantial speedups** for this package (can be easily 10x and more). Matrix factorization is already parallelized in package with OpenMP. This can be done by setting corresponding environment variables **before starting `R`**:

1. OpenBLAS: `export OPENBLAS_NUM_THREADS=1`.
1. Intel MKL: `export MKL_NUM_THREADS=1`
1. Apple Accelerate: `export VECLIB_MAXIMUM_THREADS=1`

It it also possible to change number of threads in runtime, see for example following packages:

* [OpenBlasThreads](https://github.com/rundel/OpenBlasThreads)
* [RhpcBLASctl](https://cran.r-project.org/web/packages/RhpcBLASctl/index.html)
