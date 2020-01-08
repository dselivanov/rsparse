#include "rsparse.h"
#define BPR 0
#define WARP 1
#define DOT_PRODUCT 0
#define LOGISTIC 1

const arma::uword get_random_user(const arma::uword n_user);
const arma::uword get_positive_item_index(const arma::uword max_index);
const arma::uword get_negative_candidate(const arma::uword n_item);

template <typename T> T positive_item_relevance(const MappedCSR<T> &x, const arma::uword u, const arma::uword i);
template <typename T> const arma::uvec positive_items(const MappedCSR<T> &x, const arma::uword u);

template <typename T>
T rank_loss(const T x) {
  return std::log1p(x + 1);
}

template <typename T>
T sigmoid(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

// binary_search - check if element is in the sorted array
// if yes - return true - skip sgd update
bool is_negative(const arma::uvec &arr, arma::uword x) {
  int l = 0;
  int r = arr.size();
  while (l <= r) {
    size_t m = l + (r - l) / 2;

    // Check if x is present at mid
    if (arr[m] == x)
      return false;

    // If x greater, ignore left half
    if (arr[m] < x)
      l = m + 1;

    // If x is smaller, ignore right half
    else
      r = m - 1;
  }

  // if we reach here, then element was
  // not present
  return true;
}

inline const arma::uword get_random_user(const arma::uword n_user) {
  return(arma::randi( arma::distr_param(0, n_user - 1) ));
}

inline const arma::uword get_negative_candidate(const arma::uword n_item) {
  return(arma::randi( arma::distr_param(0, n_item - 1) ));
}

template <typename T>
T positive_item_relevance(const MappedCSR<T> &x, const arma::uword u, const arma::uword i) {
  const arma::uword p1 = x.row_ptrs[u];
  return(x.values[p1 + i]);
}

template <typename T>
const arma::uvec positive_items(const MappedCSR<T> &x, const arma::uword u) {
  const arma::uword p1 = x.row_ptrs[u];
  const arma::uword p2 = x.row_ptrs[u + 1];
  arma::uvec idx = arma::uvec(&x.col_indices[p1], p2 - p1, false, true);
  return(idx);
}

const arma::uword get_positive_item_index(const arma::uword max_index) {
  arma::uword index_random_element = arma::randi( arma::distr_param(0, max_index) );
  return(index_random_element);
}

template <typename T> void warp_solver(
    const MappedCSR<T> &x,
    arma::Mat<T> &W, //user latent factors rank*n_user
    arma::Mat<T> &H, //item latent factors rank*n_item
    const arma::uword rank,
    const arma::uword n_updates,
    const T learning_rate,
    const T momentum,
    const T lambda_user,
    const T lambda_item_positive,
    const T lambda_item_negative,
    const  arma::uword n_threads,
    bool update_items = true,
    const arma::uword solver = BPR,
    const arma::uword link_function = DOT_PRODUCT,
    arma::uword max_negative_samples = 50,
    const T margin = 0.1) {


  const arma::uword TRACK = n_updates / n_threads / 10;
  const arma::uword n_user = x.n_rows;
  const arma::uword n_item = x.n_cols;
  max_negative_samples = std::min(max_negative_samples, n_item);
  const double WARP_RANK_NORMALIZER = rank_loss(double(n_item));

  // accumulated gradients for momentum
  arma::Mat<T> W_grad(W.n_rows, W.n_cols, arma::fill::zeros);
  arma::Mat<T> H_grad(H.n_rows, H.n_cols, arma::fill::zeros);

  #ifdef _OPENMP
  #pragma omp parallel num_threads(n_threads)
  #endif
  {
    size_t n_correct = 0, n_positive = 0, n_negative = 0;
    #ifdef _OPENMP
    #pragma omp for schedule(static, GRAIN_SIZE)
    #endif
    for(arma::uword iter = 0; iter < n_updates; ++iter) {
      if (n_positive >= TRACK) {
        if (is_master()) {
          Rcpp::Rcout.precision(3);
          Rcpp::Rcout << \
            100 * (iter + 1.0) / n_updates << "%" << std::setw(10) << \
              " AUC:"<< double(n_correct) / n_positive << std::setw(10) << \
              " negative_oversampling:"<< double(n_negative) / (n_positive) << std::endl;
        }
        n_correct = n_positive = n_negative = 0;
      }

      const arma::uword u = get_random_user(n_user);
      const arma::uvec idx = positive_items(x, u);

      if (idx.is_empty()) continue; // user with no positive items
      auto id = get_positive_item_index(idx.size() - 1);
      const arma::uword i = idx[id];
      n_positive++;

      const auto w_u = W.col(u);
      const auto h_i = H.col(i);
      arma::Col<T> h_j;
      arma::uword j = 0, k = 0, correct_samples = 0;
      bool skip = true;

      // hj_adjust, hi_adjust = 1 if f(w, h) = dot(w, h)
      // if f(w, h) = LOGISTIC(dot(w, h)) they are derivatives of a logistic function
      T r_ui = 1, r_uj = 1, hj_adjust = 1, hi_adjust = 1;
      double weight = 1;
      for (k = 0; k < max_negative_samples; k++) {
        j = get_negative_candidate(n_item);
        // continue if we've sampled a really negative element
        if (is_negative(idx, j)) {
          n_negative++;
          h_j = H.col(j);

          r_ui = dot(w_u, h_i);
          r_uj = dot(w_u, h_j);

          if (link_function == LOGISTIC) {
            r_ui = sigmoid(r_ui);
            r_uj = sigmoid(r_uj);
            hj_adjust = r_uj * (1 - r_uj);
            hi_adjust = r_ui * (1 - r_ui);
          }

          auto distance = r_uj - r_ui;

          if (distance < 0) {
            correct_samples++;
            // for AUC estimation
            if (k == 0) {
              n_correct += 1;
            }
          }
          weight = sigmoid<T>(distance);
          if (solver == BPR) {
            skip = false;
            break;
          } else { // WAPR loss
            // don't update on easy negatives
            if (distance + margin >= 0) {
              skip = false;
              weight *= rank_loss((double(n_item) - 1.0) / double(k + 1)) / WARP_RANK_NORMALIZER;
              break;
            }
          }
        }
      }

      if (!skip) {
        arma::Col<T> w_grad = weight * (hj_adjust * h_j - hi_adjust * h_i);
        arma::Col<T> h_grad_i = -weight * hi_adjust * w_u;
        arma::Col<T> h_grad_j = weight * hj_adjust * w_u;

        if (momentum > 0) {
          w_grad = momentum * W_grad.col(u) + (1 - momentum) * w_grad;
          h_grad_i = momentum * H_grad.col(i) + (1 - momentum) * h_grad_i;
          h_grad_j = momentum * H_grad.col(j) + (1 - momentum) * h_grad_j;
          W_grad.col(u) = w_grad;
          H_grad.col(i) = h_grad_i;
          H_grad.col(j) = h_grad_j;
        }

        W.col(u) -= learning_rate * w_grad;
        if (lambda_user > 0) {
          W.col(u) -= learning_rate * lambda_user * w_u;
        }
        if (update_items) {
          H.col(i) -=  learning_rate * h_grad_i;
          if (lambda_item_positive > 0) {
            H.col(i) -= learning_rate * lambda_item_positive * h_i;
          }
          H.col(j) -= learning_rate * h_grad_j;
          if (lambda_item_negative > 0) {
            H.col(j) -= learning_rate * lambda_item_negative * h_j;
          }
        }
      }
    }
  }
}

// [[Rcpp::export]]
void warp_solver_double(
    const Rcpp::S4 &m_csc_r,
    arma::Mat<double> &W, // user latent factors rank * n_user
    arma::Mat<double> &H, // item latent factors rank * n_item
    const arma::uword rank,
    const arma::uword n_updates,
    double learning_rate = 0.01,
    double momentum = 0.8,
    double lambda_user = 0.0,
    double lambda_item_positive = 0.0,
    double lambda_item_negative = 0.0,
    const arma::uword n_threads = 1,
    bool update_items = true,
    const arma::uword solver = 0, // BPR
    const arma::uword link_function = 0, // DOT_PRODUCT
    arma::uword max_negative_samples = 50,
    double margin = 0.1 ) {
  const dMappedCSR x = extract_mapped_csr(m_csc_r);
  warp_solver<double>(x, W, H, rank, n_updates, learning_rate, momentum,       \
                     lambda_user, lambda_item_positive, lambda_item_negative, \
                     n_threads, update_items,
                     solver, link_function,
                     max_negative_samples, margin);
}
