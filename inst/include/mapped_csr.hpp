#ifndef MAPPED_CSR_HPP
#define MAPPED_CSR_HPP

#include <cstdint>
#include <stddef.h>
#include <armadillo>

template< typename T>
class MappedCSR {
public:
  MappedCSR();
  MappedCSR(arma::uword n_rows,
            arma::uword n_cols,
            size_t nnz,
            arma::uword* col_indices,
            arma::uword* row_ptrs,
            T* values):
  n_rows(n_rows), n_cols(n_cols), nnz(nnz), col_indices(col_indices), row_ptrs(row_ptrs), values(values) {};
  const arma::uword n_rows;
  const arma::uword n_cols;
  const size_t nnz;
  arma::uword* col_indices;
  arma::uword* row_ptrs;
  T* values;
  std::pair<arma::uvec, arma::Col<T>> get_row(const arma::uword i) const {
    const arma::uword p1 = this->row_ptrs[i];
    const arma::uword p2 = this->row_ptrs[i + 1];
    const arma::uvec idx = arma::uvec(&this->col_indices[p1], p2 - p1, false, true);
    const arma::Col<T> values = arma::Col<T>(&this->values[p1], p2 - p1, false, true);
    return(std::pair<arma::uvec, arma::Col<T>>(idx, values));
  };
};

using dMappedCSR = MappedCSR<double>;
using fMappedCSR = MappedCSR<float>;

#endif /* MAPPED_CSR_HPP */
