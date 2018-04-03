# rsparse

`rsparse` is an R package for statistical learning on **sparse data**. Notably it implements many algorithms sparse **matrix factorizations** with a focus on applications for **recommender systems**.

**All of the algorithms benefit from OpenMP and most of them use BLAS**. Package scales nicely to datasets with millions of rows and millions of columns.

In order to get maximum of performance it is recommended to:

1. Use high-performance BLAS (such as OpenBLAS, MKL, Apple Accelerate).
1. Add proper compiler optimizations in your `~/.R/Makevars`. For example on recent processors (with AVX support) and complier with OpenMP support following lines could be a good option:
    ```txt
    CXX11FLAGS += -O3 -march=native -mavx -fopenmp -ffast-math
    CXXFLAGS   += -O3 -march=native -mavx -fopenmp -ffast-math
    ```
    If you are on **Mac** follow instructions [here](https://github.com/coatless/r-macos-rtools). After installation of `clang4` additionally put `PKG_CXXFLAGS += -DARMA_USE_OPENMP` line to your `~/.R/Makevars`. After that install `rsparse` in the usual way.

## Misc utils/methods

1. multithreaded `%*%` and `tcrossprod()` for `<dgRMatrix, matrix>`
1. multithreaded `%*%` and `crossprod()` for `<matrix, dgCMatrix>`

## Algorithms

### Classification/Regression

1. [Follow the proximally-regularized leader](http://www.jmlr.org/proceedings/papers/v15/mcmahan11b/mcmahan11b.pdf) which llows to solve **very large linear/logistic regression** problems with elastic-net penalty. Solver use with stochastic gradient descend with adaptive learning rates (so can be used for online learning - not necessary to load all data to RAM). See [Ad Click Prediction: a View from the Trenches](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf) for more examples.
    - Only logistic regerssion implemented at the moment
    - Native format for matrices is CSR - `Matrix::RsparseMatrix`. However common R `Matrix::CpasrseMatrix` (`dgCMatrix`) will be converted automatically.
1. [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) supervised learning algorithm which learns second order polynomial interactions in a factorized way. We provide highly optimized SIMD accelerated implementation.  

### Matrix Factorizations

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
1. Fast **Truncated SVD** and **Truncated Soft-SVD** via Alternating Least Squares as described in [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](https://arxiv.org/pdf/1410.2596.pdf). Work nice for sparse and dense matrices. Usually it is even faster than [irlba](https://github.com/bwlewis/irlba) package.
    * <img src="docs/img/soft-svd.png" width="600">
1. **Soft-Impute** via fast Alternating Least Squares as described in [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](https://arxiv.org/pdf/1410.2596.pdf).
    * <img src="docs/img/soft-impute.png" width="400">
    * with a solution in SVD form <img src="docs/img/soft-impute-svd-form.png" width="150">


## Efficiency

Here is example of `rsparse::WRMF` on [lastfm360k](https://www.upf.edu/web/mtg/lastfm360k) dataset in comparison with other good implementations:

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
