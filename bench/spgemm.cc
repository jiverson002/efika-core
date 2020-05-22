// SPDX-License-Identifier: MIT
#include <array>

#include "celero/Celero.h"

#include "efika/core.h"

#include "efika/core/blas.h"
#include "efika/core/rsb.h"
#include "efika/data/rcv1_10k.h"
//#include "efika/data/bms_pos.h"
//#include "efika/data/example.h"
//#include "efika/data/groceries.h"
//#include "efika/data/rcv1_10k.h"
//#include "efika/data/sports_1x1.h"
//#include "efika/data/youtube.h"
#include "efika/data/youtube_256.h"
//#include "efika/data/youtube_8k.h"
//#include "efika/data/youtube_10k.h"
//#include "efika/data/youtube_50k.h"

//#define DATASET         bms_pos
//#define DATASET         example
//#define DATASET         groceries
//#define DATASET         rcv1_10k
//#define DATASET         sports_1x1
//#define DATASET         youtube
#define DATASET         youtube_256
//#define DATASET         youtube_8k
//#define DATASET         youtube_10k
//#define DATASET         youtube_50k
#define xxdataset(d, v) d ## _ ## v
#define xdataset(d, v)  xxdataset(d, v)
#define dataset(v)      xdataset(DATASET, v)

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

      if (int err = EFIKA_Matrix_init(&Z1_))
        throw std::runtime_error("Could not initialize matrix Z1");

      if (int err = EFIKA_Matrix_init(&Z2_))
        throw std::runtime_error("Could not initialize matrix Z2");

      if (int err = EFIKA_Matrix_init(&C1_))
        throw std::runtime_error("Could not initialize matrix C1");

      if (int err = EFIKA_Matrix_init(&C2_))
        throw std::runtime_error("Could not initialize matrix C2");

      if (int err = EFIKA_Matrix_init(&C3_))
        throw std::runtime_error("Could not initialize matrix C3");

      if (int err = EFIKA_Matrix_init(&C4_))
        throw std::runtime_error("Could not initialize matrix C4");

      A_.mord = EFIKA_MORD_CSR;
      A_.nr   = dataset(nr);
      A_.nc   = dataset(nc);
      A_.nnz  = dataset(nnz);
      A_.ia   = dataset(ia);
      A_.ja   = dataset(ja);
      A_.a    = dataset(a);

      B1_.mord = EFIKA_MORD_CSC;
      B1_.nr   = A_.nr;
      B1_.nc   = A_.nc;
      B1_.nnz  = A_.nnz;
      B1_.ia   = A_.ia;
      B1_.ja   = A_.ja;
      B1_.a    = A_.a;

      if (int err = EFIKA_Matrix_conv(&A_, &B2_, EFIKA_MORD_CSC))
        throw std::runtime_error("Could not create inverted index B2");
      B2_.mord = EFIKA_MORD_CSR;

      if (int err = EFIKA_Matrix_conv(&A_, &Z1_, EFIKA_MORD_RSB))
        throw std::runtime_error("Could not create recursive matrix Z1");

      if (int err = EFIKA_Matrix_conv(&B2_, &Z2_, EFIKA_MORD_RSB))
        throw std::runtime_error("Could not create recursive matrix Z2");

      C1_.nr = A_.nr;
      C1_.nc = A_.nr;
      C1_.ia = static_cast<EFIKA_ind_t*>(malloc((C1_.nr + 1) * sizeof(*C1_.ia)));
      C1_.ja = static_cast<EFIKA_ind_t*>(malloc(C1_.nr * C1_.nr * sizeof(*C1_.ja)));
      C1_.a  = static_cast<EFIKA_val_t*>(malloc(C1_.nr * C1_.nr * sizeof(*C1_.a)));

      if (!(C1_.ia && C1_.ja && C1_.a))
        throw std::runtime_error("Could not allocate solution matrix C1");

      C2_.nr = A_.nr;
      C2_.nc = A_.nr;
      C2_.ia = static_cast<EFIKA_ind_t*>(malloc((C2_.nr + 1) * sizeof(*C2_.ia)));
      C2_.ja = static_cast<EFIKA_ind_t*>(malloc(C2_.nr * C2_.nr * sizeof(*C2_.ja)));
      C2_.a  = static_cast<EFIKA_val_t*>(malloc(C2_.nr * C2_.nr * sizeof(*C2_.a)));

      if (!(C2_.ia && C2_.ja && C2_.a))
        throw std::runtime_error("Could not allocate solution matrix C2");

      C3_.nr = A_.nr;
      C3_.nc = A_.nr;
      C3_.sa = static_cast<EFIKA_ind_t*>(calloc(RSB_sa_size(C3_.nr), sizeof(*C3_.ja)));
      C3_.za = static_cast<EFIKA_ind_t*>(malloc(C3_.nr * C3_.nr * sizeof(*C3_.ja)));
      C3_.a  = static_cast<EFIKA_val_t*>(malloc(C3_.nr * C3_.nr * sizeof(*C3_.a)));

      if (!(C3_.sa && C3_.za && C3_.a))
        throw std::runtime_error("Could not allocate solution matrix C3");

      ih_ = static_cast<EFIKA_ind_t*>(calloc((RSB_size(A_.nr, A_.nc) + 1), sizeof(*ih_)));
      if (!ih_)
        throw std::runtime_error("Could not allocate scratch space");

      vh_ = static_cast<EFIKA_val_t*>(calloc(std::max(A_.nr, A_.nc),
                                      sizeof(*vh_)));
      if (!vh_)
        throw std::runtime_error("Could not allocate scratch space");
    }

    void tearDown() override {
      EFIKA_Matrix_free(&B2_);
      EFIKA_Matrix_free(&Z1_);
      EFIKA_Matrix_free(&Z2_);
      EFIKA_Matrix_free(&C1_);
      EFIKA_Matrix_free(&C2_);
      EFIKA_Matrix_free(&C3_);
      EFIKA_Matrix_free(&C4_);
      free(ih_);
      free(vh_);
    }

  protected:
    EFIKA_Matrix A_;
    EFIKA_Matrix B1_;
    EFIKA_Matrix B2_;
    EFIKA_Matrix Z1_;
    EFIKA_Matrix Z2_;
    EFIKA_Matrix C1_;
    EFIKA_Matrix C2_;
    EFIKA_Matrix C3_;
    EFIKA_Matrix C4_;
    EFIKA_ind_t *ih_;
    EFIKA_val_t *vh_;
};

} // namespace

BASELINE_F(SpGEMM, CSR_CSC, SpGEMMFixture, 30, 10)
{
  efika_BLAS_spgemm_csr_csc(A_.nr, B1_.nr, A_.ia, A_.ja, A_.a, B1_.ia, B1_.ja,
                            B1_.a, C1_.ia, C1_.ja, C1_.a, vh_);
}

BENCHMARK_F(SpGEMM, CSR_CSR, SpGEMMFixture, 30, 10)
{
  efika_BLAS_spgemm_csr_csr(A_.nr, A_.ia, A_.ja, A_.a, B2_.ia, B2_.ja, B2_.a,
                            C2_.ia, C2_.ja, C2_.a, vh_);
}

BENCHMARK_F(SpGEMM, RSB_RSB, SpGEMMFixture, 30, 10)
{
  RSB_spgemm_cache_v2(RSB_size(Z1_.nr, Z1_.nc), Z1_.nnz, Z1_.za, Z1_.a, Z2_.nnz,
                      Z2_.za, Z2_.a, C3_.za, C3_.a, ih_, vh_);


}
