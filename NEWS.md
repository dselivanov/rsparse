# rsparse 0.5.1 (2022-09-11)
- update `configure` script, thanks to @david-cortes, see #73
- minor fixes in WRMF
- update docs with new roxygen2 to pass CRAN checks
- update NEWS.md ro follow CRAN format

# rsparse 0.5.0 (2021-10-17)
- reworked non-negative matrix factorization with brand-new Coordinate Descent solver for OLS
- WRMF can model user, item and global biases
- various performance improvements

# rsparse 0.4.0 (2020-04-01)
- updated docs with roxygen2 7.1
- added `ScaleNormalize` transformer
- added sparse*float S4 methods

# rsparse 0.3.3.2 (2019-07-17)
- faster `find_top_product()` - avoid BLAS and openmp thread contention
- correctly identify openmp on OSX
- fixed issue with CRAN 'rcnst' check
- use `install_name_tool` hook in the `.onLoad()` - changes location of the `float.so` for CRAN binary installation - see #25

# rsparse 0.3.3.1 (2019-04-14)
- fixed out of bound memory access as reported by CRAN UBSAN
- added ability to init GloVe embeddings with user provided values

# rsparse 0.3.3 (2019-03-16)
- added methods to natively slice CSR matrices without converting them to triplet/CSC
- add GloVe matrix factorization (adapted from `text2vec`)
- link to `float` package - credits to @snoweye and @wrathematics
