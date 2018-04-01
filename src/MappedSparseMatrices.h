#include <cstdint>
#include <stddef.h>

template< typename T>
class MappedCSR {
public:
  MappedCSR();
  MappedCSR(std::uint32_t n_rows,
            std::uint32_t n_cols,
            size_t nnz,
            std::uint32_t * col_indices,
            std::uint32_t * row_ptrs,
            T * values):
  n_rows(n_rows), n_cols(n_cols), nnz(nnz), col_indices(col_indices), row_ptrs(row_ptrs), values(values) {};
  const std::uint32_t n_rows;
  const std::uint32_t n_cols;
  const size_t nnz;
  const std::uint32_t * col_indices;
  const std::uint32_t * row_ptrs;
  T * values;
};

using dMappedCSR = MappedCSR<double>;
using fMappedCSR = MappedCSR<float>;

template< typename T>
class MappedCSC {
public:
  MappedCSC();
  MappedCSC(std::uint32_t n_rows,
            std::uint32_t n_cols,
            size_t nnz,
            std::uint32_t * row_indices,
            std::uint32_t * col_ptrs,
            T * values):
    n_rows(n_rows), n_cols(n_cols), nnz(nnz), row_indices(row_indices), col_ptrs(col_ptrs), values(values) {};
  const std::uint32_t n_rows;
  const std::uint32_t n_cols;
  const size_t nnz;
  const std::uint32_t * row_indices;
  const std::uint32_t * col_ptrs;
  T * values;
};

using dMappedCSC = MappedCSC<double>;
using fMappedCSC = MappedCSC<float>;
