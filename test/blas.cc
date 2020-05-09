// SPDX-License-Identifier: MIT
#include <algorithm>
#include <array>
#include <stdexcept>

#include <gtest/gtest.h>

#include "efika/core.h"

#include "efika/core/blas.h"
#include "efika/data/rcv1_10k.h"

namespace {

class BLAS : public ::testing::Test {
  public:
    void SetUp() override {
      if (int err = EFIKA_Matrix_init(&A_))
        throw std::runtime_error("Could not initialize matrix A");

      if (int err = EFIKA_Matrix_init(&B1_))
        throw std::runtime_error("Could not initialize matrix B1");

      if (int err = EFIKA_Matrix_init(&B2_))
        throw std::runtime_error("Could not initialize matrix B2");

      if (int err = EFIKA_Matrix_init(&Z_))
        throw std::runtime_error("Could not initialize matrix Z");

      if (int err = EFIKA_Matrix_init(&C1_))
        throw std::runtime_error("Could not initialize matrix C1");

      if (int err = EFIKA_Matrix_init(&C2_))
        throw std::runtime_error("Could not initialize matrix C2");

      if (int err = EFIKA_Matrix_init(&C3_))
        throw std::runtime_error("Could not initialize matrix C3");

      A_.mord = EFIKA_MORD_CSR;
      A_.nr   = rcv1_10k_nr;
      A_.nc   = rcv1_10k_nc;
      A_.nnz  = rcv1_10k_nnz;
      A_.ia   = rcv1_10k_ia;
      A_.ja   = rcv1_10k_ja;
      A_.a    = rcv1_10k_a;

      B1_.mord = EFIKA_MORD_CSC;
      B1_.nr   = rcv1_10k_nr;
      B1_.nc   = rcv1_10k_nc;
      B1_.nnz  = rcv1_10k_nnz;
      B1_.ia   = rcv1_10k_ia;
      B1_.ja   = rcv1_10k_ja;
      B1_.a    = rcv1_10k_a;

      if (int err = EFIKA_Matrix_conv(&A_, &B2_, EFIKA_MORD_CSC))
        throw std::runtime_error("Could not create inverted index B2");
      B2_.mord = EFIKA_MORD_CSR;

      if (int err = EFIKA_Matrix_conv(&A_, &Z_, EFIKA_MORD_RSB))
        throw std::runtime_error("Could not create inverted index Z");

      C1_.nr = rcv1_10k_nr;
      C1_.nc = rcv1_10k_nc;
      C1_.ia = static_cast<EFIKA_ind_t*>(malloc((C1_.nr + 1) * sizeof(*C1_.ia)));
      C1_.ja = static_cast<EFIKA_ind_t*>(malloc(C1_.nr * C1_.nr * sizeof(*C1_.ja)));
      C1_.a  = static_cast<EFIKA_val_t*>(malloc(C1_.nr * C1_.nr * sizeof(*C1_.a)));

      if (!(C1_.ia && C1_.ja && C1_.a))
        throw std::runtime_error("Could not allocate solution matrix C1");

      C2_.nr = rcv1_10k_nr;
      C2_.nc = rcv1_10k_nc;
      C2_.ia = static_cast<EFIKA_ind_t*>(malloc((C2_.nr + 1) * sizeof(*C2_.ia)));
      C2_.ja = static_cast<EFIKA_ind_t*>(malloc(C2_.nr * C2_.nr * sizeof(*C2_.ja)));
      C2_.a  = static_cast<EFIKA_val_t*>(malloc(C2_.nr * C2_.nr * sizeof(*C2_.a)));

      if (!(C2_.ia && C2_.ja && C2_.a))
        throw std::runtime_error("Could not allocate solution matrix C2");

      h_ = static_cast<EFIKA_val_t*>(calloc(std::max(rcv1_10k_nr, rcv1_10k_nc),
                                     sizeof(*h_)));

      if (!h_)
        throw std::runtime_error("Could not allocate scratch space");
    }

    void TearDown() override {
      EFIKA_Matrix_free(&B2_);
      EFIKA_Matrix_free(&Z_);
      EFIKA_Matrix_free(&C1_);
      EFIKA_Matrix_free(&C2_);
      EFIKA_Matrix_free(&C3_);
      free(h_);
    }

  protected:
    EFIKA_Matrix A_;
    EFIKA_Matrix B1_;
    EFIKA_Matrix B2_;
    EFIKA_Matrix Z_;
    EFIKA_Matrix C1_;
    EFIKA_Matrix C2_;
    EFIKA_Matrix C3_;
    EFIKA_val_t *h_;
};

} // namespace

TEST_F(BLAS, SpGEMM) {
  efika_BLAS_spgemm_csr_csc(A_.nr, B1_.nr, A_.ia, A_.ja, A_.a, B1_.ia, B1_.ja,
                            B1_.a, C1_.ia, C1_.ja, C1_.a, h_);

  efika_BLAS_spgemm_csr_csr(A_.nr, A_.ia, A_.ja, A_.a, B2_.ia, B2_.ja, B2_.a,
                            C2_.ia, C2_.ja, C2_.a, h_);

  //const auto n = next_pow2(std::max(Z.nr, Z.nc));

  //efika_BLAS_spgemm_csr_csr(n, Z_.nnz, Z_.sa, Z_.za, Z_.a, Z_.nnz, Z_.sa, Z_.za,
  //                          Z_.a, ...);

  ASSERT_EQ(C1_.nr, C2_.nr);
  ASSERT_EQ(C1_.nc, C2_.nc);
  for (EFIKA_ind_t i = 0; i <= C1_.nr; i++)
    ASSERT_EQ(C1_.ia[i], C2_.ia[i]) << "i = " << i;
  for (EFIKA_ind_t i = 1; i <= C1_.nr; i++)
    ASSERT_GE(C1_.ia[i], C1_.ia[i - 1]);

  for (EFIKA_ind_t i = 0; i < C1_.nr; i++) {
    std::vector<std::pair<EFIKA_ind_t, EFIKA_val_t>> kv1;
    std::vector<std::pair<EFIKA_ind_t, EFIKA_val_t>> kv2;

    for (EFIKA_ind_t j = C1_.ia[i]; j < C1_.ia[i + 1]; j++)
      kv1.push_back({ C1_.ja[j], C1_.a[j] });

    for (EFIKA_ind_t j = C2_.ia[i]; j < C2_.ia[i + 1]; j++)
      kv2.push_back({ C2_.ja[j], C2_.a[j] });

    sort(kv1.begin(), kv1.end(), [](const auto & a, const auto & b) -> bool {
      return a.first > b.first;
    });
    sort(kv2.begin(), kv2.end(), [](const auto & a, const auto & b) -> bool {
      return a.first > b.first;
    });

    const auto rnnz = C1_.ia[i + 1] - C1_.ia[i];

    for (EFIKA_ind_t j = 0; j < rnnz; j++)
      ASSERT_EQ(kv1[j].first, kv2[j].first) << "(i,j) = (" << i << ", " << kv1[j].first << ")";

    for (EFIKA_ind_t j = 0; j < rnnz; j++)
      ASSERT_EQ(kv1[j].second, kv2[j].second) << "(i,j) = (" << i << ", " << kv1[j].first << ")";
  }
}
