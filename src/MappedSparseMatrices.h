#include <cstdint>
#include <stddef.h>
#include <RcppArmadillo.h>
template< typename T>
class MappedCSR {
public:
  MappedCSR();
  MappedCSR(arma::uword n_rows,
            arma::uword n_cols,
            size_t nnz,
            arma::uword * col_indices,
            arma::uword * row_ptrs,
            T * values):
  n_rows(n_rows), n_cols(n_cols), nnz(nnz), col_indices(col_indices), row_ptrs(row_ptrs), values(values) {};
  const arma::uword n_rows;
  const arma::uword n_cols;
  const size_t nnz;
  const arma::uword * col_indices;
  const arma::uword * row_ptrs;
  T * values;
};

using dMappedCSR = MappedCSR<double>;
using fMappedCSR = MappedCSR<float>;

template< typename T>
class MappedCSC {
public:
  MappedCSC();
  MappedCSC(arma::uword n_rows,
            arma::uword n_cols,
            size_t nnz,
            arma::uword * row_indices,
            arma::uword * col_ptrs,
            T * values):
    n_rows(n_rows), n_cols(n_cols), nnz(nnz), row_indices(row_indices), col_ptrs(col_ptrs), values(values) {};
  const arma::uword n_rows;
  const arma::uword n_cols;
  const size_t nnz;
  const arma::uword * row_indices;
  const arma::uword * col_ptrs;
  T * values;
};

using dMappedCSC = MappedCSC<double>;
using fMappedCSC = MappedCSC<float>;
