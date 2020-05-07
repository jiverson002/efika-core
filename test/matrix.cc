// SPDX-License-Identifier: MIT
#include <array>

#include <gtest/gtest.h>

#include "efika/core.h"
#include "efika/data/rcv1_10k.h"

namespace {

class Matrix : public ::testing::Test {
  public:
    void SetUp() override {
      if (int err = EFIKA_Matrix_init(&A_))
        throw std::runtime_error("Could not initialize matrix A");

      A_.nr  = rcv1_10k_nr;
      A_.nc  = rcv1_10k_nc;
      A_.nnz = rcv1_10k_nnz;
      A_.ia  = rcv1_10k_ia;
      A_.ja  = rcv1_10k_ja;
      A_.a   = rcv1_10k_a;

      if (int err = EFIKA_Matrix_sort(&A_, EFIKA_COL | EFIKA_ASC))
        throw std::runtime_error("Could not sort rows of matrix A");
    }

    void TearDown() override {
    }

  protected:
    EFIKA_Matrix A_;
};

} // namespace

TEST_F(Matrix, IIDX) {
  EFIKA_Matrix I, B;

  int err = EFIKA_Matrix_init(&I);
  ASSERT_EQ(0, err);
  err = EFIKA_Matrix_init(&B);
  ASSERT_EQ(0, err);

  err = EFIKA_Matrix_iidx(&A_, &I);
  ASSERT_EQ(0, err);
  err = EFIKA_Matrix_iidx(&I, &B);
  ASSERT_EQ(0, err);

  ASSERT_EQ(A_.nr, B.nr);
  ASSERT_EQ(A_.nc, B.nc);
  ASSERT_EQ(A_.nnz, B.nnz);
  for (EFIKA_ind_t i = 0; i <= A_.nr; i++)
    ASSERT_EQ(A_.ia[i], B.ia[i]) << "i = " << i;
  for (EFIKA_ind_t i = 0; i < A_.nnz; i++)
    ASSERT_EQ(A_.ja[i], B.ja[i]) << "i = " << i;
  for (EFIKA_ind_t i = 0; i < A_.nnz; i++)
    ASSERT_EQ(A_.a[i], B.a[i]) << "i = " << i;

  EFIKA_Matrix_free(&I);
  EFIKA_Matrix_free(&B);
}

TEST_F(Matrix, toRSB) {
  EFIKA_Matrix Z;

  int err = EFIKA_Matrix_init(&Z);
  ASSERT_EQ(0, err);

  err = EFIKA_Matrix_rsb(&A_, &Z);
  ASSERT_EQ(0, err);

  EFIKA_Matrix_free(&Z);
}
