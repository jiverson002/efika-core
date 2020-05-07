// SPDX-License-Identifier: MIT
#include <array>
#include <cstdlib>

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

  auto check_leaf = [](
    EFIKA_ind_t const ro,
    EFIKA_ind_t const co,
    EFIKA_ind_t const nr,
    EFIKA_ind_t const nc,
    EFIKA_ind_t const nnz,
    EFIKA_ind_t const * const za,
    EFIKA_val_t const * const a
  ) -> bool {
    // TODO: do a more thorough check of the actual non-zero values, searching
    //       the rows of A_ to make sure that non-zero values found in Z are
    //       correct according to A_.

    std::size_t const half = sizeof(EFIKA_ind_t) * CHAR_BIT / 2;
    EFIKA_ind_t const mask = ((EFIKA_ind_t)-1) >> half;
    for (EFIKA_ind_t k = 0; k < nnz; k++) {
      if ((za[k] >> half) >= nr) return false;
      if ((za[k] &  mask) >= nc) return false;
    }
    return true;
  };

  std::function<bool(
    EFIKA_ind_t const ro,
    EFIKA_ind_t const co,
    EFIKA_ind_t const nr,
    EFIKA_ind_t const nc,
    EFIKA_ind_t const nnz,
    EFIKA_ind_t const * const sa,
    EFIKA_ind_t const * const za,
    EFIKA_val_t const * const a
  )> check_node;

  check_node = [&](
    EFIKA_ind_t const ro,
    EFIKA_ind_t const co,
    EFIKA_ind_t const nr,
    EFIKA_ind_t const nc,
    EFIKA_ind_t const nnz,
    EFIKA_ind_t const * const sa,
    EFIKA_ind_t const * const za,
    EFIKA_val_t const * const a
  ) -> bool {
    #define RSB_MIN_NODE_SIZE 1024

    /* don't split node if it is too small */
    if (nnz < RSB_MIN_NODE_SIZE)
      return check_leaf(ro, co, nr, nc, nnz, za, a);

    /* compute row and column split keys */
    const auto rsp = ro + nr / 2;
    const auto csp = co + nc / 2;

    /* compute quadrant dimensions */
    const auto nnr = nr / 2;
    const auto nnc = nc / 2;

    /* compute quadrant offsets */
    const auto bh = ro + nnr;
    const auto rh = co + nnc;

    /* compute quadrant # non-zeros */
    const auto nnz0 = sa[0];
    const auto nnz1 = sa[1] - sa[0];
    const auto nnz2 = sa[2] - sa[1];
    const auto nnz3 = nnz   - sa[2];

    /* recursively check each quadrant */
    return check_node(ro, co, nnr, nnc, nnz0, sa + 6, za, a)
      && check_node(ro, rh, nnr, nnc, nnz1, sa + sa[3], za + sa[0], a + sa[0])
      && check_node(bh, co, nnr, nnc, nnz2, sa + sa[4], za + sa[1], a + sa[1])
      && check_node(bh, rh, nnr, nnc, nnz3, sa + sa[5], za + sa[2], a + sa[2]);

    #undef RSB_MIN_NODE_SIZE
  };

  // FIXME: nr and nc must be powers of 2
  //bool const valid = check_node(0, 0, Z.nr, Z.nc, Z.nnz, Z.sa, Z.za, Z.a);
  bool const valid = check_node(0, 0, 65536, 65536, Z.nnz, Z.sa, Z.za, Z.a);

  ASSERT_TRUE(valid);

  EFIKA_Matrix_free(&Z);
}
