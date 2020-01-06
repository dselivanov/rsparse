#include "rsparse.h"

const arma::uword get_random_user(const arma::uword n_user);
const arma::uword get_positive_item_index(const arma::uword max_index);
const arma::uword get_negative_candidate(const arma::uword n_item);

template <typename T> T positive_item_relevance(const MappedCSR<T> &x, const arma::uword u, const arma::uword i);
template <typename T> const arma::uvec positive_items(const MappedCSR<T> &x, const arma::uword u);

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

template <typename T>
T rank_loss(const T x) {
  return std::log1p(x + 1);
}

template <typename T> void bpr_solver(
    const MappedCSR<T> &x,
    arma::Mat<T> &W, //user latent factors rank*n_user
    arma::Mat<T> &H,  //item latent factors rank*n_item
    const arma::uword rank,
    const arma::uword n_updates,
    const T learning_rate,
    const T momentum,
    const T lambda_user,
    const T lambda_item_positive,
    const T lambda_item_negative,
    const  arma::uword n_threads,
    bool update_items = true
    ) {


  const arma::uword TRACK = n_updates / n_threads / 10;
  const arma::uword n_user = x.n_rows;
  const arma::uword n_item = x.n_cols;
  // max_negative_samples = std::min(max_negative_samples, n_item);
  // const double WARP_RANK_NORMALIZER = rank_loss(double(n_item));

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

      // T relevance = positive_item_relevance(x, u, id);
      if (n_positive >= TRACK) {
        if (is_master()) {
          Rcpp::Rcout.precision(3);
          Rcpp::Rcout << \
            100 * (iter + 1.0) / n_updates << "%" << std::setw(10) << \
            " AUC:"<< double(n_correct) / (n_negative) << std::endl;
        }
        n_correct = n_positive = n_negative = 0;
      }

      const arma::uword u = get_random_user(n_user);
      const arma::uvec idx = positive_items(x, u);

      if (idx.is_empty()) continue; // user with no positive items
      auto id = get_positive_item_index(idx.size() - 1);
      const arma::uword i = idx[id];
      n_positive++;

      // T diff = 0, score = 0;
      // arma::uword j = get_negative_candidate(n_item);
      // arma::Col<T> h_j;
      // bool skip = false;
      // const double weight = 1;
      // if (is_negative(idx, j)) {
      //   n_negative++;
      //   h_j = H.col(j);
      //   diff = dot(w_u, h_i) - dot(w_u, h_j);
      //   score = 1.0 / (1.0 + std::exp(diff));
      //   if (score < .5) {
      //     n_correct++;
      //   }
      // } else {
      //   skip = true;
      // }

      // arma::Col<T> h_j;
      //
      // arma::uword j = 0, k = 0;
      // bool skip = true;
      // T diff = 0, score = 0;
      // double weight = 0;
      // for (k = 0; k < max_negative_samples; k++) {
      //   j = get_negative_candidate(n_item);
      //   // continue if we've sampled a really negative element
      //   if (is_negative(idx, j)) {
      //     n_negative++;
      //     h_j = H.col(j);
      //     diff = dot(w_u, h_i) - dot(w_u, h_j);
      //     score = 1.0 / (1.0 + std::exp(diff));
      //     if (score < .5) {
      //       n_correct++;
      //     }
      //     // if (score > thresh) {
      //     //   skip = false;
      //     // }
      //   }
      //   if(!skip) {
      //     // WARP weight
      //     weight = rank_loss((double(n_item) - 1.0) / double(k + 1)) / WARP_RANK_NORMALIZER;
      //     break;
      //   }
      // }

      arma::uword j = get_negative_candidate(n_item);
      if(is_negative(idx, j)) {
        const auto w_u = W.col(u);
        const auto h_i = H.col(i);
        const auto h_j = H.col(j);

        n_negative++;
        const double weight = 1;
        arma::Col<T> w_grad = weight * (h_i - h_j);
        arma::Col<T> h_grad_i =  weight * w_u;
        arma::Col<T> h_grad_j = -weight * w_u;
        auto score = 1.0 / (1.0 + std::exp(dot(w_u, h_i) - dot(w_u, h_j)));
        if (score < .5) n_correct++;

        if (momentum > 0) {
          w_grad = momentum * W_grad.col(u) + (1 - momentum) * w_grad;
          h_grad_i = momentum * H_grad.col(i) + (1 - momentum) * h_grad_i;
          h_grad_j = momentum * H_grad.col(j) + (1 - momentum) * h_grad_j;
          W_grad.col(u) = w_grad;
          H_grad.col(i) = h_grad_i;
          H_grad.col(j) = h_grad_j;
        }

        W.col(u) += learning_rate * score * w_grad;
        if (lambda_user > 0) {
          W.col(u) += learning_rate * lambda_user * w_u;
        }
        if (update_items) {
          H.col(i) += learning_rate * score * h_grad_i;
          if (lambda_item_positive > 0) {
            H.col(i) += learning_rate * lambda_item_positive * h_i;
          }
          H.col(j) += learning_rate * score * h_grad_j;
          if (lambda_item_negative > 0) {
            H.col(j) += learning_rate * lambda_item_negative * h_j;
          }
        }
      }
    }
  }
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



// [[Rcpp::export]]
void bpr_solver_double(
    const Rcpp::S4 &m_csc_r,
    arma::Mat<double> &W, //user latent factors rank*n_user
    arma::Mat<double> &H,  //item latent factors rank*n_item
    const arma::uword rank = 8,
    const arma::uword n_updates = 1e5,
    double learning_rate = 0.01,
    double momentum = 0.8,
    double lambda_user = 0.0,
    double lambda_item_positive = 0.0,
    double lambda_item_negative = 0.0,
    const arma::uword n_threads = 1,
    bool update_items = true
) {
  const dMappedCSR x = extract_mapped_csr(m_csc_r);
  bpr_solver<double>(x, W, H, rank, n_updates, learning_rate, momentum, \
                     lambda_user, lambda_item_positive, lambda_item_negative, \
                     n_threads, update_items);
}


/*
 library("rsparse")
 data("movielens100k")
 x = as(movielens100k, "RsparseMatrix")
 rank = 32
 n_updates = 1e6
 learning_rate = 0.1
 momentum = 0.8
 lambda_user = lambda_item_positive = lambda_item_negative = 0
 n_threads = 6
 update_items = TRUE
 n_users = nrow(x)
 n_items = ncol(x)
 W = matrix(runif(rank * n_users), ncol = n_users)
 H = matrix(runif(rank * n_items), ncol = n_items)
 system.time(rsparse:::bpr_solver_double(x, W, H, rank, n_updates, learning_rate, lambda_user, lambda_item_positive, lambda_item_negative, n_threads, update_items))
 */
