#ifndef MAPPED_CSC_HPP
#define MAPPED_CSC_HPP

#include <stddef.h>
#include <armadillo>
#include <cstdint>

template <typename T>
class MappedCSC {
 public:
  MappedCSC();
  MappedCSC(arma::uword n_rows, arma::uword n_cols, size_t nnz, arma::uword* row_indices,
            arma::uword* col_ptrs, T* values)
      : n_rows(n_rows),
        n_cols(n_cols),
        nnz(nnz),
        row_indices(row_indices),
        col_ptrs(col_ptrs),
        values(values){};
  const arma::uword n_rows;
  const arma::uword n_cols;
  const size_t nnz;
  arma::uword* row_indices;
  arma::uword* col_ptrs;
  T* values;
};

using dMappedCSC = MappedCSC<double>;
using fMappedCSC = MappedCSC<float>;

#endif /* MAPPED_CSC_HPP */
