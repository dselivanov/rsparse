<a href="https://www.rexy.ai"><img src="https://www.rexy.ai/images/logo.png" height="64" width="64"></a>. 
[![Travis-CI Build Status](https://travis-ci.org/dselivanov/rsparse.svg?branch=master)](https://travis-ci.org/dselivanov/rsparse)
[![codecov](https://codecov.io/gh/dselivanov/rsparse/branch/master/graph/badge.svg)](https://codecov.io/gh/dselivanov/rsparse/branch/master)
[![License](https://eddelbuettel.github.io/badges/GPL2+.svg)](http://www.gnu.org/licenses/gpl-2.0.html)
[![Project Status](https://img.shields.io/badge/lifecycle-maturing-blue.svg)](https://www.tidyverse.org/lifecycle/#maturing)


`rsparse` is an R package for statistical learning primarily on **sparse matrices** -  **matrix factorizations, factorization machines, out-of-core regression**. Many of the implemented algorithms are particularly useful for **recommender systems** and **NLP**. 

On top of that we provide some optimized routines to work on sparse matrices - multithreaded <dense, sparse> matrix multiplications and improved support for sparse matrices in CSR format (`Matrix::RsparseMatrix`).

We've paid some attention to the implementation details - we try to avoid data copies, utilize multiple threads via OpenMP and use SIMD where appropriate. Package **allows to work on datasets with millions of rows and millions of columns**.


### Support 

Please reach us if you need **commercial support** - [hello@rexy.ai](mailto:hello@rexy.ai).



# Features

### Classification/Regression

1. [Follow the proximally-regularized leader](http://www.jmlr.org/proceedings/papers/v15/mcmahan11b/mcmahan11b.pdf) which llows to solve **very large linear/logistic regression** problems with elastic-net penalty. Solver use with stochastic gradient descend with adaptive learning rates (so can be used for online learning - not necessary to load all data to RAM). See [Ad Click Prediction: a View from the Trenches](https://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf) for more examples.
    - Only logistic regerssion implemented at the moment
    - Native format for matrices is CSR - `Matrix::RsparseMatrix`. However common R `Matrix::CpasrseMatrix` (`dgCMatrix`) will be converted automatically.
1. [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) supervised learning algorithm which learns second order polynomial interactions in a factorized way. We provide highly optimized SIMD accelerated implementation.  

### Matrix Factorizations

1. Vanilla **Maximum Margin Matrix Factorization** - classic approch for "rating" prediction. See `WRMF` class and constructor option `feedback = "explicit"`. Original paper which indroduced MMMF could be found [here](http://ttic.uchicago.edu/~nati/Publications/MMMFnips04.pdf).
    * <img src="https://raw.githubusercontent.com/dselivanov/rsparse/master/docs/img/MMMF.png" width="400">
1. **Weighted Regularized Matrix Factorization (WRMF)** from [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf). See `WRMF` class and constructor option `feedback = "implicit"`. 
We provide 2 solvers:
    1. Exact based of Cholesky Factorization
    1. Approximated based on fixed number of steps of **Conjugate Gradient**.
See details in [Applications of the Conjugate Gradient Method for Implicit Feedback Collaborative Filtering](https://pdfs.semanticscholar.org/bfdf/7af6cf7fd7bb5e6b6db5bbd91be11597eaf0.pdf) and [Faster Implicit Matrix Factorization](www.benfrederickson.com/fast-implicit-matrix-factorization/).
    * <img src="https://raw.githubusercontent.com/dselivanov/rsparse/master/docs/img/WRMF.png" width="400">
1. **Linear-Flow** from [Practical Linear Models for Large-Scale One-Class Collaborative Filtering](http://www.bkveton.com/https://raw.githubusercontent.com/dselivanov/rsparse/master/docs/ijcai2016.pdf). Algorithm looks for factorized low-rank item-item similarity matrix (in some sense it is similar to [SLIM](http://glaros.dtc.umn.edu/gkhome/node/774))
    * <img src="https://raw.githubusercontent.com/dselivanov/rsparse/master/docs/img/LinearFlow.png" width="300">
1. Fast **Truncated SVD** and **Truncated Soft-SVD** via Alternating Least Squares as described in [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](https://arxiv.org/pdf/1410.2596.pdf). Works for both sparse and dense matrices. Works on [float](https://github.com/wrathematics/float) matrices as well! For certain problems may be even faster than [irlba](https://github.com/bwlewis/irlba) package.
    * <img src="https://raw.githubusercontent.com/dselivanov/rsparse/master/docs/img/soft-svd.png" width="600">
1. **Soft-Impute** via fast Alternating Least Squares as described in [Matrix Completion and Low-Rank SVD via Fast Alternating Least Squares](https://arxiv.org/pdf/1410.2596.pdf).
    * <img src="https://raw.githubusercontent.com/dselivanov/rsparse/master/docs/img/soft-impute.png" width="400">
    * with a solution in SVD form <img src="https://raw.githubusercontent.com/dselivanov/rsparse/master/docs/img/soft-impute-svd-form.png" width="150">
1. **GloVe** as described in [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf).
    * This is usually used to train word embeddings, but actually also very useful for recommender systems.

### Optimized matrix operations

1. multithreaded `%*%` and `tcrossprod()` for `<dgRMatrix, matrix>`
1. multithreaded `%*%` and `crossprod()` for `<matrix, dgCMatrix>`
1. natively slice `CSR` matrices (`Matrix::RsparseMatrix`) without converting them to triplet / CSC

# Installation 

Most of the algorithms benefit from OpenMP and many of them could utilize high-performance implementation of BLAS. If you want make maximum out of the package please read the section below carefuly.

It is recommended to:

1. Use high-performance BLAS (such as OpenBLAS, MKL, Apple Accelerate).
1. Add proper compiler optimizations in your `~/.R/Makevars`. For example on recent processors (with AVX support) and complier with OpenMP support following lines could be a good option:
    ```txt
    CXX11FLAGS += -O3 -march=native -mavx -fopenmp -ffast-math
    CXXFLAGS   += -O3 -march=native -mavx -fopenmp -ffast-math
    ```

If you are on **Mac** follow instructions [here](https://github.com/coatless/r-macos-rtools). After installation of `clang4` additionally put `PKG_CXXFLAGS += -DARMA_USE_OPENMP` line to your `~/.R/Makevars`. After that install `rsparse` in a usual way.


# Materials

**Note that syntax is these posts/slides is not up to date since package was under active development**

1. [Slides from DataFest Tbilisi(2017-11-16)](https://www.slideshare.net/DmitriySelivanov/matrix-factorizations-for-recommender-systems)
1. [Introduction to matrix factorization with Weighted-ALS algorithm](http://dsnotes.com/post/2017-05-28-matrix-factorization-for-recommender-systems/) - collaborative filtering for implicit feedback datasets.
1. [Music recommendations using LastFM-360K dataset](http://dsnotes.com/post/2017-06-28-matrix-factorization-for-recommender-systems-part-2/)
    * evaluation metrics for ranking
    * setting up proper cross-validation
    * possible issues with nested parallelism and thread contention
    * making recommendations for new users
    * complimentary item-to-item recommendations
1. [Benchmark](http://dsnotes.com/post/2017-07-10-bench-wrmf/) against other good implementations

Here is example of `rsparse::WRMF` on [lastfm360k](https://www.upf.edu/web/mtg/lastfm360k) dataset in comparison with other good implementations:

<img src="https://github.com/dselivanov/bench-wals/raw/master/img/wals-bench-cg.png" width="600">


# API

We follow [mlapi](https://github.com/dselivanov/mlapi) conventions.

# Configure

Generate configure:

```sh
autoconf configure.ac > configure && chmod +x configure
```
