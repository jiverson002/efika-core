// SPDX-License-Identifier: MIT
#include <array>

#include <gtest/gtest.h>

#include "efika/core.h"

namespace {

class Matrix : public ::testing::Test {
  public:
    void SetUp() override {
      M_.nr  = nr_;
      M_.nc  = nc_;
      M_.nnz = nnz_;
      M_.ia = m_ia_.data();
      M_.ja = m_ja_.data();
      M_.a  = m_a_.data();
    }

    void TearDown() override {
    }

  protected:
    EFIKA_ind_t const nr_  { 4 };
    EFIKA_ind_t const nc_  { 8 };
    EFIKA_ind_t const nnz_ { 8 };
    std::array<EFIKA_ind_t, 5> m_ia_ { 0, 1, 2, 5, 8 };
    std::array<EFIKA_ind_t, 8> m_ja_ { 0, 1, 1, 3, 5, 0, 3, 7 };
    std::array<EFIKA_val_t, 8> m_a_  { 0, 1, 2, 3, 4, 5, 6, 7 };
    std::array<EFIKA_ind_t, 8> z_za_ {
      0x00000000 /* r0c0 */, 0x00010001 /* r1c1 */, 0x00020001 /* r2c1 */,
      0x00030000 /* r3c0 */, 0x00020003 /* r2c3 */, 0x00030003 /* r3c3 */,
      0x00020005 /* r2c5 */, 0x00030007 /* r3c7 */
    };
    std::array<EFIKA_val_t, 8> z_a_  { 0, 1, 2, 5, 3, 6, 4, 7 };
    EFIKA_Matrix M_;
};

} // namespace

TEST_F(Matrix, IIDX) {
  EFIKA_Matrix I, B;

  int err = EFIKA_Matrix_init(&I);
  ASSERT_EQ(0, err);
  err = EFIKA_Matrix_init(&B);
  ASSERT_EQ(0, err);

  err = EFIKA_Matrix_iidx(&M_, &I);
  ASSERT_EQ(0, err);
  err = EFIKA_Matrix_iidx(&I, &B);
  ASSERT_EQ(0, err);

  ASSERT_EQ(this->M_.nr, B.nr);
  ASSERT_EQ(this->M_.nc, B.nc);
  ASSERT_EQ(this->M_.nnz, B.nnz);
  for (EFIKA_ind_t i = 0; i <= this->M_.nr; i++)
    ASSERT_EQ(this->M_.ia[i], B.ia[i]) << "i = " << i;
  for (EFIKA_ind_t i = 0; i < this->M_.nnz; i++)
    ASSERT_EQ(this->M_.ja[i], B.ja[i]) << "i = " << i;
  for (EFIKA_ind_t i = 0; i < this->M_.nnz; i++)
    ASSERT_EQ(this->M_.a[i], B.a[i]) << "i = " << i;

  EFIKA_Matrix_free(&I);
  EFIKA_Matrix_free(&B);
}

TEST_F(Matrix, toRSB) {
  EFIKA_Matrix Z;

  int err = EFIKA_Matrix_init(&Z);
  ASSERT_EQ(0, err);

  err = EFIKA_Matrix_rsb(&this->M_, &Z);
  ASSERT_EQ(0, err);

  for (int i = 0; i < this->m_a_.size(); i++) {
    ASSERT_EQ(this->z_za_[i],  Z.za[i]) << "i = " << i;
    ASSERT_EQ(this->z_a_[i],  Z.a[i]) << "i = " << i;
  }

  EFIKA_Matrix_free(&Z);
}
