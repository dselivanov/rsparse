#include "rsparse.h"
#define TRACK 100000

arma::uword get_random_user(const arma::uword n_user);
arma::uword get_positive_item(const arma::uvec &idx);
arma::uword get_negative_item(const arma::uword n_item);
//arma::uword get_negative_item(const dMappedCSR &x, const arma::uword u);
arma::uvec positive_items(const dMappedCSR &x, const arma::uword u);

// binary_search - check if element is in the sorted array
// if yes - return true - skip sgd update
bool skip_element(const arma::uvec &arr, arma::uword x) {
  int l = 0;
  int r = arr.size();
  while (l <= r) {
    size_t m = l + (r - l) / 2;

    // Check if x is present at mid
    if (arr[m] == x)
      return true;

    // If x greater, ignore left half
    if (arr[m] < x)
      l = m + 1;

    // If x is smaller, ignore right half
    else
      r = m - 1;
  }

  // if we reach here, then element was
  // not present
  return false;
}

template <typename T> void bpr_solver(
    const dMappedCSR &x,
    arma::Mat<T> &W, //user latent factors rank*n_user
    arma::Mat<T> &H,  //item latent factors rank*n_item
    const arma::uword rank,
    const arma::uword n_updates,
    T learning_rate,
    // T momentum,
    T lambda_user,
    T lambda_item_positive,
    T lambda_item_negative,
    const  arma::uword n_threads,
    bool update_items = true
    ) {

  const arma::uword n_user = x.n_rows;
  const arma::uword n_item = x.n_cols;

  // // accumulated gradients for momentum
  // arma::Mat<T> W_grad(W.n_rows, W.n_cols, arma::fill::zeros);
  // arma::Mat<T> H_grad(H.n_rows, H.n_cols, arma::fill::zeros);

  #ifdef _OPENMP
  #pragma omp parallel num_threads(n_threads)
  #endif
  {
    size_t n_correct = 0, n_done = 0, n_skip = 0;
    #ifdef _OPENMP
    #pragma omp for schedule(guided, GRAIN_SIZE)
    #endif
    for(arma::uword iter = 0; iter < n_updates; ++iter) {
      const arma::uword u = get_random_user(n_user);
      const arma::uvec idx = positive_items(x, u);
      const arma::uword i = get_positive_item(idx);
      const arma::uword j = get_negative_item(n_item);

      n_done++;
      if (n_done % TRACK == TRACK - 1) {
        #pragma omp critical
        {
          std::cout << "step " << iter + 1 << "/" << n_updates << " AUC:"<< (n_correct + 0.0) / (TRACK - n_skip) << " skip " << n_skip << std::endl;
        }
        n_correct = 0;
        n_skip = 0;
        n_done = 0;
      }

      bool skip = skip_element(idx, j);
      // bool skip = false;
      if(skip) {
        n_skip++;
      } else {
        const auto w_u = W.col(u);
        const auto h_i = H.col(i);
        const auto h_j = H.col(j);

        T score = 1.0 / (1.0 + std::exp(dot(w_u, h_i) - dot(w_u, h_j)));
        if (score < .5) n_correct++;

        // W_grad.col(u) = momentum * W_grad.col(u) + (1 - momentum) * (H.col(i) - H.col(j));
        // H_grad.col(i) = momentum * H_grad.col(i) + (1 - momentum) * W.col(u);
        // H_grad.col(j) = momentum * H_grad.col(j) - (1 - momentum) * W.col(u);

        W.col(u) += learning_rate * score * (h_i - h_j);
        if (lambda_user > 0) {
          W.col(u) += learning_rate * lambda_user * w_u;
        }
        if (update_items) {
          const auto h_grad = learning_rate * score * w_u;
          H.col(i) += h_grad;
          if (lambda_item_positive > 0) {
            H.col(i) += learning_rate * lambda_item_positive * h_i;
          }
          H.col(j) -= h_grad;
          if (lambda_item_negative > 0) {
            H.col(j) += learning_rate * lambda_item_negative * h_j;
          }
        }
      }
    }
  }
}

inline arma::uword get_random_user(const arma::uword n_user) {
  return(arma::randi( arma::distr_param(0, n_user - 1) ));
}

inline arma::uword get_negative_item(const arma::uword n_item) {
  return(arma::randi( arma::distr_param(0, n_item - 1) ));
}

arma::uvec positive_items(const dMappedCSR &x, const arma::uword u) {
  const arma::uword p1 = x.row_ptrs[u];
  const arma::uword p2 = x.row_ptrs[u + 1];
  arma::uvec idx = arma::uvec(&x.col_indices[p1], p2 - p1, false, true);
  return(idx);
}

inline arma::uword get_positive_item(const arma::uvec &idx) {
  arma::uword index_random_element = arma::randi( arma::distr_param(0, idx.size() - 1) );
  return(idx[index_random_element]);
}



// [[Rcpp::export]]
void bpr_solver_double(
    const Rcpp::S4 &m_csc_r,
    arma::Mat<double> &W, //user latent factors rank*n_user
    arma::Mat<double> &H,  //item latent factors rank*n_item
    const arma::uword rank = 8,
    const arma::uword n_updates = 1e5,
    double learning_rate = 0.01,
    double lambda_user = 0.0,
    double lambda_item_positive = 0.0,
    double lambda_item_negative = 0.0,
    const arma::uword n_threads = 1,
    bool update_items = true
) {
  const dMappedCSR x = extract_mapped_csr(m_csc_r);
  bpr_solver<double>(x, W, H, rank, n_updates, learning_rate, lambda_user, lambda_item_positive, lambda_item_negative, n_threads, update_items);
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
