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

class SpGEMM : public ::testing::Test {
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

TEST_F(SpGEMM, RSB_RSB_CANARY) {
  std::array<EFIKA_ind_t, 6> za11 {
    0x00000000 /* r0c0 */, 0x00010001 /* r1c1 */, 0x00020001 /* r2c1 */,
    0x00030000 /* r3c0 */, 0x00020003 /* r2c3 */, 0x00030003 /* r3c3 */
  };
  std::array<EFIKA_val_t, 6> a11 { 1, 1, 2, 5, 3, 6 };

  std::array<EFIKA_ind_t, 2> za12 {
    0x00020005 /* r2c5 */, 0x00030007 /* r3c7 */
  };
  std::array<EFIKA_val_t, 2> a12 { 4, 7 };

  std::array<EFIKA_ind_t, 6> zb11 {
    0x00000000 /* r0c0 */, 0x00010001 /* r1c1 */, 0x00000003 /* r0c3 */,
    0x00010002 /* r1c2 */, 0x00030002 /* r3c2 */, 0x00030003 /* r3c3 */
  };
  std::array<EFIKA_val_t, 6> b11 { 1, 1, 5, 2, 3, 6 };

  std::array<EFIKA_ind_t, 2> zb12 {
    0x00050002 /* r5c2 */, 0x00070003 /* r7c3 */
  };
  std::array<EFIKA_val_t, 2> b12 { 4, 7 };

  std::array<EFIKA_ind_t, 20> ih;
  std::array<EFIKA_val_t, 20> vh;

  std::fill(vh.begin(), vh.end(), 0.0);

  std::array<EFIKA_ind_t, 11> zc1;
  std::array<EFIKA_val_t, 10> c1;
  std::array<EFIKA_ind_t,  2> zc2;
  std::array<EFIKA_val_t,  2> c2;
  std::array<EFIKA_ind_t, 12> zc;
  std::array<EFIKA_val_t, 12> c;

  std::array<EFIKA_ind_t, 10> zd1 {
    0x00000000 /* r0c0 */, 0x00000003 /* r0c3 */, 0x00010001 /* r1c1 */,
    0x00010002 /* r1c2 */, 0x00020001 /* r2c1 */, 0x00020002 /* r2c2 */,
    0x00020003 /* r2c3 */, 0x00030000 /* r3c0 */, 0x00030003 /* r3c3 */,
    0x00030002 /* r3c2 */
  };
  std::array<EFIKA_val_t, 10> d1 { 1.0, 5.0, 1.0, 2.0, 2.0, 13.0, 18.0, 5.0,
                                   61.0, 18.0 };

  std::array<EFIKA_ind_t, 2> zd2 {
    0x00020002 /* r2c2 */, 0x00030003 /* r3c3 */
  };
  std::array<EFIKA_val_t, 2> d2 { 16.0, 49.0 };

  std::array<EFIKA_ind_t, 10> zd {
    0x00000000 /* r0c0 */, 0x00000003 /* r0c3 */, 0x00010001 /* r1c1 */,
    0x00010002 /* r1c2 */, 0x00020002 /* r2c2 */, 0x00020001 /* r2c1 */,
    0x00020003 /* r2c3 */, 0x00030003 /* r3c3 */, 0x00030000 /* r3c0 */,
    0x00030002 /* r3c2 */
  };
  std::array<EFIKA_val_t, 10> d { 1.0, 5.0, 1.0, 2.0, 29.0, 2.0, 18.0, 110.0,
                                  5.0, 18.0 };

  const auto nnz1 = RSB_spgemm_cache(4,
                                     za11.size(), za11.data(), a11.data(),
                                     zb11.size(), zb11.data(), b11.data(),
                                     zc1.data(), c1.data(),
                                     ih.data(), vh.data());

  const auto nnz2 = RSB_spgemm_cache(4,
                                     za12.size(), za12.data(), a12.data(),
                                     zb12.size(), zb12.data(), b12.data(),
                                     zc2.data(), c2.data(),
                                     ih.data(), vh.data());

  const auto cnnz = RSB_spgemm_merge(4,
                                     nnz1, zc1.data(), c1.data(),
                                     nnz2, zc2.data(), c2.data(),
                                     zc.data(), c.data(), vh.data());

  ASSERT_EQ(nnz1, zd1.size());
  for (EFIKA_ind_t i = 0; i < zd1.size(); i++) {
    ASSERT_EQ(zc1[i], zd1[i]) << "i = " << i
      << " (" << RSB_row(zc1[i]) << ", " << RSB_col(zc1[i]) << ") <=>"
      << " (" << RSB_row(zd1[i]) << ", " << RSB_col(zd1[i]) << ")";
    ASSERT_EQ(c1[i], d1[i]) << "i = " << i
      << " (" << RSB_row(zc1[i]) << ", " << RSB_col(zc1[i]) << ") <=>"
      << " (" << RSB_row(zd1[i]) << ", " << RSB_col(zd1[i]) << ")";
  }

  ASSERT_EQ(nnz2, zd2.size());
  for (EFIKA_ind_t i = 0; i < zd2.size(); i++) {
    ASSERT_EQ(zc2[i], zd2[i]) << "i = " << i
      << " (" << RSB_row(zc2[i]) << ", " << RSB_col(zc2[i]) << ") <=>"
      << " (" << RSB_row(zd2[i]) << ", " << RSB_col(zd2[i]) << ")";
    ASSERT_EQ(c2[i], d2[i]) << "i = " << i
      << " (" << RSB_row(zc2[i]) << ", " << RSB_col(zc2[i]) << ") <=>"
      << " (" << RSB_row(zd2[i]) << ", " << RSB_col(zd2[i]) << ")";
  }

  ASSERT_EQ(cnnz, zd.size());
  for (EFIKA_ind_t i = 0; i < zd.size(); i++) {
    ASSERT_EQ(zc[i], zd[i]) << "i = " << i
      << " (" << RSB_row(zc[i]) << ", " << RSB_col(zc[i]) << ") <=>"
      << " (" << RSB_row(zd[i]) << ", " << RSB_col(zd[i]) << ")";
    ASSERT_EQ(c[i], d[i]) << "i = " << i
      << " (" << RSB_row(zc[i]) << ", " << RSB_col(zc[i]) << ") <=>"
      << " (" << RSB_row(zd[i]) << ", " << RSB_col(zd[i]) << ")";
  }
}

TEST_F(SpGEMM, RSB_RSB) {
  efika_BLAS_spgemm_csr_csc(A_.nr, B1_.nr, A_.ia, A_.ja, A_.a, B1_.ia, B1_.ja,
                            B1_.a, C1_.ia, C1_.ja, C1_.a, vh_);


  C3_.nnz = RSB_spgemm_cache(RSB_size(Z1_.nr, Z1_.nc),
                             Z1_.nnz, Z1_.za, Z1_.a,
                             Z2_.nnz, Z2_.za, Z2_.a,
                             C3_.za, C3_.a,
                             ih_, vh_);

  C3_.mord = EFIKA_MORD_RSB;
  if (int err = EFIKA_Matrix_conv(&C3_, &C4_, EFIKA_MORD_CSR))
    throw std::runtime_error("Could not convert C3 to C4");

  ASSERT_EQ(C1_.nr, C4_.nr);
  ASSERT_EQ(C1_.nc, C4_.nc);
  for (EFIKA_ind_t i = 0; i <= C1_.nr; i++)
    ASSERT_EQ(C1_.ia[i], C4_.ia[i]) << "i = " << i;

  for (EFIKA_ind_t i = 0; i < C1_.nr; i++) {
    std::vector<std::pair<EFIKA_ind_t, EFIKA_val_t>> kv1;
    std::vector<std::pair<EFIKA_ind_t, EFIKA_val_t>> kv2;
    std::vector<std::pair<EFIKA_ind_t, EFIKA_val_t>> kv4;

    for (EFIKA_ind_t j = C1_.ia[i]; j < C1_.ia[i + 1]; j++) {
      kv1.push_back({ C1_.ja[j], C1_.a[j] });
      kv4.push_back({ C4_.ja[j], C4_.a[j] });
    }

    sort(kv1.begin(), kv1.end(), [](const auto & a, const auto & b) -> bool {
      return a.first > b.first;
    });
    sort(kv4.begin(), kv4.end(), [](const auto & a, const auto & b) -> bool {
      return a.first > b.first;
    });

    const auto rnnz = C1_.ia[i + 1] - C1_.ia[i];

    for (EFIKA_ind_t j = 0; j < rnnz; j++)
      ASSERT_EQ(kv1[j].first, kv4[j].first);

    for (EFIKA_ind_t j = 0; j < rnnz; j++)
      ASSERT_EQ(kv1[j].second, kv4[j].second);
  }
}
