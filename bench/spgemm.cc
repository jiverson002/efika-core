// SPDX-License-Identifier: MIT
#include <array>

#include "celero/Celero.h"

#include "efika/core.h"

#include "efika/core/blas.h"
#include "efika/data/rcv1_10k.h"

namespace {

class SpGEMMFixture : public ::celero::TestFixture {
  public:
    void setUp(const ::celero::TestFixture::ExperimentValue&) override {
      if (int err = EFIKA_Matrix_init(&A_))
        throw std::runtime_error("Could not initialize matrix A");

      if (int err = EFIKA_Matrix_init(&B1_))
        throw std::runtime_error("Could not initialize matrix B1");

      if (int err = EFIKA_Matrix_init(&B2_))
        throw std::runtime_error("Could not initialize matrix B2");

      if (int err = EFIKA_Matrix_init(&C1_))
        throw std::runtime_error("Could not initialize matrix C1");

      if (int err = EFIKA_Matrix_init(&C2_))
        throw std::runtime_error("Could not initialize matrix C2");

      A_.nr  = rcv1_10k_nr;
      A_.nc  = rcv1_10k_nc;
      A_.nnz = rcv1_10k_nnz;
      A_.ia  = rcv1_10k_ia;
      A_.ja  = rcv1_10k_ja;
      A_.a   = rcv1_10k_a;

      B1_.nr  = rcv1_10k_nr;
      B1_.nc  = rcv1_10k_nc;
      B1_.nnz = rcv1_10k_nnz;
      B1_.ia  = rcv1_10k_ia;
      B1_.ja  = rcv1_10k_ja;
      B1_.a   = rcv1_10k_a;

      if (int err = EFIKA_Matrix_iidx(&A_, &B2_))
        throw std::runtime_error("Could not create inverted index B2");

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

      h_ = static_cast<EFIKA_val_t*>(malloc(std::max(rcv1_10k_nr, rcv1_10k_nc) * sizeof(*h_)));

      if (!h_)
        throw std::runtime_error("Could not allocate scratch space");
    }

    void tearDown() override {
      EFIKA_Matrix_free(&B2_);
      EFIKA_Matrix_free(&C1_);
      EFIKA_Matrix_free(&C2_);
      free(h_);
    }

  protected:
    EFIKA_Matrix A_;
    EFIKA_Matrix B1_;
    EFIKA_Matrix B2_;
    EFIKA_Matrix C1_;
    EFIKA_Matrix C2_;
    EFIKA_val_t *h_;
};

} // namespace

BASELINE_F(SpGEMM, CSR_CSC, SpGEMMFixture, 3, 1)
{
  efika_BLAS_spgemm_csr_csc(A_.nr, B1_.nr, A_.ia, A_.ja, A_.a, B1_.ia, B1_.ja,
                            B1_.a, C1_.ia, C1_.ja, C1_.a, h_);
}

BENCHMARK_F(SpGEMM, CSR_IDX, SpGEMMFixture, 3, 1)
{
  efika_BLAS_spgemm_csr_idx(A_.nr, A_.ia, A_.ja, A_.a, B2_.ia, B2_.ja, B2_.a,
                            C2_.ia, C2_.ja, C2_.a, h_);
}

BENCHMARK_F(SpGEMM, RSB_RSB, SpGEMMFixture, 3, 1)
{
  celero::DoNotOptimizeAway(0);
}
