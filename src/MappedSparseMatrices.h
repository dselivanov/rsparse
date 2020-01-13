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
            arma::uword * col_indices,
            arma::uword * row_ptrs,
            T * values):
  n_rows(n_rows), n_cols(n_cols), nnz(nnz), col_indices(col_indices), row_ptrs(row_ptrs), values(values) {};
  const arma::uword n_rows;
  const arma::uword n_cols;
  const size_t nnz;
  arma::uword * col_indices;
  arma::uword * row_ptrs;
  T * values;
  const std::pair<const arma::uvec, const arma::Col<T>> get_row(const arma::uword i) const {
    const arma::uword p1 = this->row_ptrs[i];
    const arma::uword p2 = this->row_ptrs[i + 1];
    const arma::uvec idx = arma::uvec(&this->col_indices[p1], p2 - p1, false, true);
    const arma::Col<T> values = arma::Col<T>(&this->values[p1], p2 - p1, false, true);
    return(std::pair<const arma::uvec, const arma::Col<T>>(idx, values));
  };
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
  arma::uword * row_indices;
  arma::uword * col_ptrs;
  T * values;
};

using dMappedCSC = MappedCSC<double>;
using fMappedCSC = MappedCSC<float>;
