// SPDX-License-Identifier: MIT
#include <algorithm>
#include <array>
#include <stdexcept>

#include <gtest/gtest.h>

#include "efika/core.h"

#include "efika/core/blas.h"
#include "efika/core/rsb.h"
//#include "efika/data/bms_pos.h"
//#include "efika/data/example.h"
//#include "efika/data/groceries.h"
//#include "efika/data/rcv1_10k.h"
//#include "efika/data/sports_1x1.h"
//#include "efika/data/youtube.h"
//#include "efika/data/youtube_8k.h"
#include "efika/data/youtube_10k.h"
//#include "efika/data/youtube_50k.h"

//#define DATASET         bms_pos
//#define DATASET         example
//#define DATASET         groceries
//#define DATASET         rcv1_10k
//#define DATASET         sports_1x1
//#define DATASET         youtube
//#define DATASET         youtube_8k
#define DATASET         youtube_10k
//#define DATASET         youtube_50k
#define xxdataset(d, v) d ## _ ## v
#define xdataset(d, v)  xxdataset(d, v)
#define dataset(v)      xdataset(DATASET, v)

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


      //for (EFIKA_ind_t i = 0; i < 67108864; i++) {
      for (EFIKA_ind_t i = 0; i < 100000000; i++) {
      //for (EFIKA_ind_t i = 0; i < 3000000000; i++) {
        C3_.za[i] = (EFIKA_ind_t)-1;
        C3_.a[i]  = 0.0;
      }

      ih_ = static_cast<EFIKA_ind_t*>(calloc((RSB_size(A_.nr, A_.nc) + 1), sizeof(*ih_)));
      if (!ih_)
        throw std::runtime_error("Could not allocate scratch space");

      vh_ = static_cast<EFIKA_val_t*>(calloc(std::max(A_.nr, A_.nc),
                                      sizeof(*vh_)));
      if (!vh_)
        throw std::runtime_error("Could not allocate scratch space");
    }

    void TearDown() override {
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

TEST_F(BLAS, SpGEMM_RSB_RSB_CANARY) {
  std::array<EFIKA_ind_t, 8> a_za {
    0x00000000 /* r0c0 */, 0x00010001 /* r1c1 */, 0x00020001 /* r2c1 */,
    0x00030000 /* r3c0 */, 0x00020003 /* r2c3 */, 0x00030003 /* r3c3 */,
    0x00020005 /* r2c5 */, 0x00030007 /* r3c7 */
  };
  std::array<EFIKA_val_t, 8> a_a { 1, 1, 2, 5, 3, 6, 4, 7 };

  std::array<EFIKA_ind_t, 8> b_za {
    0x00000000 /* r0c0 */, 0x00010001 /* r1c1 */, 0x00000003 /* r0c3 */,
    0x00010002 /* r1c2 */, 0x00030002 /* r3c2 */, 0x00030003 /* r3c3 */,
    0x00050002 /* r5c2 */, 0x00070003 /* r7c3 */
  };
  std::array<EFIKA_val_t, 8> b_a { 1, 1, 5, 2, 3, 6, 4, 7 };

  std::array<EFIKA_ind_t, 10> d_za {
    0x00000000 /* r0c0 */, 0x00000003 /* r0c3 */, 0x00010001 /* r1c1 */,
    0x00010002 /* r1c2 */, 0x00020001 /* r2c1 */, 0x00020002 /* r2c2 */,
    0x00020003 /* r2c3 */, 0x00030000 /* r3c0 */, 0x00030002 /* r3c2 */,
    0x00030003 /* r3c3 */
  };
  std::array<EFIKA_val_t, 10> d_a { 2.0, 10.0, 2.0, 4.0, 4.0, 58.0, 36.0, 10.0,
                                    36.0, 220.0 };

  std::array<EFIKA_ind_t, 10> c_za;
  std::array<EFIKA_val_t, 10> c_a;
  std::array<EFIKA_ind_t, 20> ih;

  for (EFIKA_ind_t i = 0; i < c_za.size(); i++) {
    c_za[i] = (EFIKA_ind_t)-1;
    c_a[i]  = 0.0;
  }

  ind_t c_nnz;

  efika_BLAS_spgemm_rsb_rsb(8,
                            8, NULL, a_za.data(), a_a.data(),
                            8, NULL, b_za.data(), b_a.data(),
                            &c_nnz, NULL, c_za.data(), c_a.data(),
                            ih.data());

  efika_BLAS_spgemm_rsb_rsb(8,
                            8, NULL, a_za.data(), a_a.data(),
                            8, NULL, b_za.data(), b_a.data(),
                            &c_nnz, NULL, c_za.data(), c_a.data(),
                            ih.data());

  for (EFIKA_ind_t i = 0; i < c_za.size(); i++) {
    ASSERT_EQ(c_za[i], d_za[i]);
    ASSERT_EQ(c_a[i], d_a[i]);
  }
}

TEST_F(BLAS, SpGEMM) {
  efika_BLAS_spgemm_csr_csc(A_.nr, B1_.nr, A_.ia, A_.ja, A_.a, B1_.ia, B1_.ja,
                            B1_.a, C1_.ia, C1_.ja, C1_.a, vh_);

  efika_BLAS_spgemm_csr_csr(A_.nr, A_.ia, A_.ja, A_.a, B2_.ia, B2_.ja, B2_.a,
                            C2_.ia, C2_.ja, C2_.a, vh_);

  efika_BLAS_spgemm_rsb_rsb(RSB_size(Z1_.nr, Z1_.nc), Z1_.nnz, Z1_.sa, Z1_.za,
                            Z1_.a, Z2_.nnz, Z2_.sa, Z2_.za, Z2_.a, &C3_.nnz,
                            C3_.sa, C3_.za, C3_.a, ih_);

  C3_.mord = EFIKA_MORD_RSB;
  if (int err = EFIKA_Matrix_conv(&C3_, &C4_, EFIKA_MORD_CSR))
    throw std::runtime_error("Could not convert C3 to C4");

  std::cerr << ">>> " << C1_.ia[C1_.nr] << std::endl;
  std::cerr << ">>> " << C4_.ia[C4_.nr] << std::endl;

  ASSERT_EQ(C1_.nr, C2_.nr);
  ASSERT_EQ(C1_.nc, C2_.nc);
  ASSERT_EQ(C1_.nr, C4_.nr);
  ASSERT_EQ(C1_.nc, C4_.nc);
  for (EFIKA_ind_t i = 0; i <= C1_.nr; i++) {
    ASSERT_EQ(C1_.ia[i], C2_.ia[i]) << "i = " << i;
    ASSERT_EQ(C1_.ia[i], C4_.ia[i]) << "i = " << i;
  }

  for (EFIKA_ind_t i = 0; i < C1_.nr; i++) {
    std::vector<std::pair<EFIKA_ind_t, EFIKA_val_t>> kv1;
    std::vector<std::pair<EFIKA_ind_t, EFIKA_val_t>> kv2;
    std::vector<std::pair<EFIKA_ind_t, EFIKA_val_t>> kv4;

    for (EFIKA_ind_t j = C1_.ia[i]; j < C1_.ia[i + 1]; j++) {
      kv1.push_back({ C1_.ja[j], C1_.a[j] });
      kv2.push_back({ C2_.ja[j], C2_.a[j] });
      kv4.push_back({ C4_.ja[j], C4_.a[j] });
    }

    sort(kv1.begin(), kv1.end(), [](const auto & a, const auto & b) -> bool {
      return a.first > b.first;
    });
    sort(kv2.begin(), kv2.end(), [](const auto & a, const auto & b) -> bool {
      return a.first > b.first;
    });
    sort(kv4.begin(), kv4.end(), [](const auto & a, const auto & b) -> bool {
      return a.first > b.first;
    });

    const auto rnnz = C1_.ia[i + 1] - C1_.ia[i];

    for (EFIKA_ind_t j = 0; j < rnnz; j++) {
      ASSERT_EQ(kv1[j].first, kv2[j].first);
      ASSERT_EQ(kv1[j].first, kv4[j].first);
    }

    for (EFIKA_ind_t j = 0; j < rnnz; j++) {
      ASSERT_EQ(kv1[j].second, kv2[j].second);
      ASSERT_EQ(kv1[j].second, kv4[j].second);
    }
  }
}
