#include <cstdint>
#include <stddef.h>

template< typename T>
class MappedCSR {
public:
  MappedCSR();
  MappedCSR(std::uint32_t n_rows,
            std::uint32_t n_cols,
            size_t nnz,
            std::uint32_t * j,
            std::uint32_t * p,
            T * x):
  n_rows(n_rows), n_cols(n_cols), nnz(nnz), j(j), p(p), x(x) {};
  const std::uint32_t n_rows;
  const std::uint32_t n_cols;
  const size_t nnz;
  const std::uint32_t * j;
  const std::uint32_t * p;
  T * x;
};

using dMappedCSR = MappedCSR<double>;
using fMappedCSR = MappedCSR<float>;
