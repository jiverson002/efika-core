// SPDX-License-Identifier: MIT
#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>

#include <gtest/gtest.h>

#include "efika/core.h"
#include "efika/data/rcv1_10k.h"

namespace {

class Matrix : public ::testing::Test {
  public:
    void SetUp() override {
      if (int err = EFIKA_Matrix_init(&A_))
        throw std::runtime_error("Could not initialize matrix A");

      A_.mord = EFIKA_MORD_CSR;
      A_.nr  = rcv1_10k_nr;
      A_.nc  = rcv1_10k_nc;
      A_.nnz = rcv1_10k_nnz;
      A_.ia = static_cast<EFIKA_ind_t*>(std::malloc((A_.nr + 1) * sizeof(EFIKA_ind_t)));
      A_.ja = static_cast<EFIKA_ind_t*>(std::malloc(A_.nnz * sizeof(EFIKA_ind_t)));
      A_.a  = static_cast<EFIKA_val_t*>(std::malloc(A_.nnz * sizeof(EFIKA_val_t)));

      if (!(A_.ia && A_.ja && A_.a))
        throw std::runtime_error("Could not allocate memory for matrix A");

      std::copy(rcv1_10k_ia, rcv1_10k_ia + rcv1_10k_nr + 1, A_.ia);
      std::copy(rcv1_10k_ja, rcv1_10k_ja + rcv1_10k_nnz, A_.ja);
      std::copy(rcv1_10k_a, rcv1_10k_a + rcv1_10k_nnz, A_.a);

      /* reorder columns in decreasing order of degree */
      if (int err = EFIKA_Matrix_cord(&A_, EFIKA_DEG | EFIKA_DSC))
        throw std::runtime_error("Could not reorder columns of matrix A");

      /* reorder rows in decreasing order of row maximum */
      if (int err = EFIKA_Matrix_rord(&A_, EFIKA_VAL | EFIKA_DSC))
        throw std::runtime_error("Could not reorder rows of matrix A");

      /* reorder each row in increasing order of column id */
      if (int err = EFIKA_Matrix_sort(&A_, EFIKA_COL | EFIKA_ASC))
        throw std::runtime_error("Could not sort rows of matrix A");
    }

    void TearDown() override {
      EFIKA_Matrix_free(&A_);
    }

  protected:
    EFIKA_Matrix A_;
};

} // namespace

TEST_F(Matrix, toCSC) {
  EFIKA_Matrix I, B;

  int err = EFIKA_Matrix_init(&I);
  ASSERT_EQ(0, err);
  err = EFIKA_Matrix_init(&B);
  ASSERT_EQ(0, err);

  err = EFIKA_Matrix_conv(&A_, &I, EFIKA_MORD_CSC);
  ASSERT_EQ(0, err);
  std::cout << I.nc << std::endl;
  err = EFIKA_Matrix_conv(&I, &B, EFIKA_MORD_CSR);
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

  err = EFIKA_Matrix_conv(&A_, &Z, EFIKA_MORD_RSB);
  ASSERT_EQ(0, err);

  auto check_leaf = [this](
    const EFIKA_ind_t ro,
    const EFIKA_ind_t co,
    const EFIKA_ind_t n,
    const EFIKA_ind_t nnz,
    const EFIKA_ind_t * const za,
    const EFIKA_val_t * const a
  ) -> bool {
    for (EFIKA_ind_t k = 0; k < nnz; k++) {
      const auto h = sizeof(EFIKA_ind_t) * CHAR_BIT / 2;
      const auto m = ((EFIKA_ind_t)-1) >> h;
      const auto r = (za[k] >> h);
      const auto c = (za[k] &  m);

      if (r >= n) return false;
      if (c >= n) return false;

      /* compensate for compressed leaf indexing */
      const auto gr = ro + r;
      const auto gc = co + c;

      /* find the global column in the original matrix row */
      const auto p = std::find(A_.ja + A_.ia[gr], A_.ja + A_.ia[gr + 1], gc);

      /* make sure that it was found */
      if (*p != gc) return false;

      /* find the index in the original ja array for the column */
      const auto i = A_.ia[gr] + (p - (A_.ja + A_.ia[gr]));

      /* make sure the non-zero value was stored correctly */
      if (a[k] != A_.a[i]) return false;
    }

    return true;
  };

  std::function<bool(
    const EFIKA_ind_t ro,
    const EFIKA_ind_t co,
    const EFIKA_ind_t n,
    const EFIKA_ind_t nnz,
    const EFIKA_ind_t * const sa,
    const EFIKA_ind_t * const za,
    const EFIKA_val_t * const a
  )> check_node;

  check_node = [&check_node, &check_leaf](
    const EFIKA_ind_t ro,
    const EFIKA_ind_t co,
    const EFIKA_ind_t n,
    const EFIKA_ind_t nnz,
    const EFIKA_ind_t * const sa,
    const EFIKA_ind_t * const za,
    const EFIKA_val_t * const a
  ) -> bool {
    /* don't explicitly split node if dimensions are small enough */
    if (n <= (EFIKA_ind_t)1 << sizeof(EFIKA_ind_t) * CHAR_BIT / 2)
      return check_leaf(ro, co, n, nnz, za, a);

    /* compute quadrant dimensions */
    const auto nn = n / 2;

    /* compute quadrant # non-zeros */
    const auto nnz0 = sa[0];
    const auto nnz1 = sa[1] - sa[0];
    const auto nnz2 = sa[2] - sa[1];
    const auto nnz3 = nnz   - sa[2];

    /* recursively check each quadrant */
    return check_node(ro, co, nn, nnz0, sa + 6, za, a)
        && check_node(ro, co + nn, nn, nnz1, sa + sa[3], za + sa[0], a + sa[0])
        && check_node(ro + nn, co, nn, nnz2, sa + sa[4], za + sa[1], a + sa[1])
        && check_node(ro + nn, co + nn, nn, nnz3, sa + sa[5], za + sa[2], a + sa[2]);
  };

  // FIXME: nr and nc must be powers of 2
  bool const valid = check_node(0, 0, 65536, Z.nnz, Z.sa, Z.za, Z.a);

  ASSERT_TRUE(valid);

  EFIKA_Matrix_free(&Z);
}
