#include <cstdint>
#include <stddef.h>
#include <RcppArmadillo.h>
#include <RcppEigen.h>

using namespace Rcpp;

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

static dMappedCSR extract_mapped_csr(S4 input) {
  IntegerVector dim = input.slot("Dim");
  NumericVector rx = input.slot("x");
  uint32_t nrows = dim[0];
  uint32_t ncols = dim[1];
  IntegerVector rj = input.slot("j");
  IntegerVector rp = input.slot("p");
  return dMappedCSR(nrows, ncols, rx.length(), (uint32_t *)rj.begin(), (uint32_t *)rp.begin(), rx.begin());
}
