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

class SpGEMM : public ::testing::Test {
  public:
    void SetUp() override {
      if (int err = EFIKA_Matrix_init(&A_))
        throw std::runtime_error("Could not initialize matrix A");

      if (int err = EFIKA_Matrix_init(&B_))
        throw std::runtime_error("Could not initialize matrix B");

      if (int err = EFIKA_Matrix_init(&C_))
        throw std::runtime_error("Could not initialize matrix C");

      A_.mord = EFIKA_MORD_CSR;
      A_.nr   = dataset(nr);
      A_.nc   = dataset(nc);
      A_.nnz  = dataset(nnz);
      A_.ia   = dataset(ia);
      A_.ja   = dataset(ja);
      A_.a    = dataset(a);

      if (int err = EFIKA_Matrix_conv(&A_, &B_, EFIKA_MORD_RSB))
        throw std::runtime_error("Could not create recursive matrix B");

      C_.nr = A_.nr;
      C_.nc = A_.nr;
      C_.za = static_cast<EFIKA_ind_t*>(malloc(C_.nr * C_.nr * sizeof(*C_.ja)));
      C_.a  = static_cast<EFIKA_val_t*>(malloc(C_.nr * C_.nr * sizeof(*C_.a)));

      if (!(C_.za && C_.a))
        throw std::runtime_error("Could not allocate solution matrix C");

      ih_ = static_cast<EFIKA_ind_t*>(malloc((A_.nr + 1) * sizeof(*ih_)));
      if (!ih_)
        throw std::runtime_error("Could not allocate index scratch space");

      vh_ = static_cast<EFIKA_val_t*>(malloc(std::max(A_.nr, A_.nc) * sizeof(*vh_)));
      if (!vh_)
        throw std::runtime_error("Could not allocate value scratch space");
    }

    void TearDown() override {
      EFIKA_Matrix_free(&B_);
      EFIKA_Matrix_free(&C_);
      free(ih_);
      free(vh_);
    }

  protected:
    EFIKA_Matrix A_;
    EFIKA_Matrix B_;
    EFIKA_Matrix C_;
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

#if 0
  std::array<EFIKA_ind_t, 11> zc;
  std::array<EFIKA_val_t, 10> c;

  std::array<EFIKA_ind_t, 10> zd {
    0x00000000 /* r0c0 */, 0x00000003 /* r0c3 */, 0x00010001 /* r1c1 */,
    0x00010002 /* r1c2 */, 0x00020001 /* r2c1 */, 0x00020002 /* r2c2 */,
    0x00020003 /* r2c3 */, 0x00030000 /* r3c0 */, 0x00030003 /* r3c3 */,
    0x00030002 /* r3c2 */
  };
  std::array<EFIKA_val_t, 10> d { 1.0, 5.0, 1.0, 2.0, 2.0, 29.0, 18.0, 5.0,
                                  110.0, 18.0 };

  auto cnnz = RSB_spgemm_cache_v2(4,
                                  za11.size(), za11.data(), a11.data(),
                                  zb11.size(), zb11.data(), b11.data(),
                                  0, zc.data(), c.data(),
                                  ih.data(), vh.data());

  cnnz = RSB_spgemm_cache_v2(4,
                             za12.size(), za12.data(), a12.data(),
                             zb12.size(), zb12.data(), b12.data(),
                             cnnz, zc.data(), c.data(),
                             ih.data(), vh.data());

  ASSERT_EQ(cnnz, zd.size());
  for (EFIKA_ind_t i = 0; i < zd.size(); i++) {
    ASSERT_EQ(zc[i], zd[i]) << "i = " << i
      << " (" << RSB_row(zc[i]) << ", " << RSB_col(zc[i]) << ") <=>"
      << " (" << RSB_row(zd[i]) << ", " << RSB_col(zd[i]) << ")";
    ASSERT_EQ(c[i], d[i]) << "i = " << i
      << " (" << RSB_row(zc[i]) << ", " << RSB_col(zc[i]) << ") <=>"
      << " (" << RSB_row(zd[i]) << ", " << RSB_col(zd[i]) << ")";
  }
#else
  std::array<EFIKA_ind_t, 11> zc1;
  std::array<EFIKA_val_t, 10> c1;
  std::array<EFIKA_ind_t,  3> zc2;
  std::array<EFIKA_val_t,  2> c2;

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

  const auto nnz1 = RSB_spgemm_cache_v2(4,
                                        za11.size(), za11.data(), a11.data(),
                                        zb11.size(), zb11.data(), b11.data(),
                                        zc1.data(), c1.data(),
                                        ih.data(), vh.data());

  const auto nnz2 = RSB_spgemm_cache_v2(4,
                                        za12.size(), za12.data(), a12.data(),
                                        zb12.size(), zb12.data(), b12.data(),
                                        zc2.data(), c2.data(),
                                        ih.data(), vh.data());

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
#endif
}

#if 0
TEST_F(SpGEMM, SpGEMM) {
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
#endif
