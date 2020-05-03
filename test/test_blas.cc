// SPDX-License-Identifier: MIT
#include <array>
#include <iostream>
#include <stdexcept>

#include <gtest/gtest.h>

#include "efika/core.h"

#include "efika/core/blas.h"
//#include "efika/data/rcv1_10k.h"

namespace {

class BLAS : public ::testing::Test {
  public:
    void SetUp() override {
      A_.nr  = nr_;
      A_.nc  = nc_;
      A_.nnz = nnz_;
      A_.ia = A_ia_.data();
      A_.ja = A_ja_.data();
      A_.a  = A_a_.data();
      B1_.nr  = nr_;
      B1_.nc  = nc_;
      B1_.nnz = nnz_;
      B1_.ia = B1_ia_.data();
      B1_.ja = B1_ja_.data();
      B1_.a  = B1_a_.data();
      B2_.nr  = nc_;
      B2_.nc  = nr_;
      B2_.nnz = nnz_;
      B2_.ia = B2_ia_.data();
      B2_.ja = B2_ja_.data();
      B2_.a  = B2_a_.data();
      C1_.nr  = nr_;
      C1_.nc  = nr_;
      C1_.nnz = C1_a_.size();
      C1_.ia = C1_ia_.data();
      C1_.ja = C1_ja_.data();
      C1_.a  = C1_a_.data();
      C2_.nr  = nr_;
      C2_.nc  = nr_;
      C2_.nnz = C2_a_.size();
      C2_.ia = C2_ia_.data();
      C2_.ja = C2_ja_.data();
      C2_.a  = C2_a_.data();
    }

    void TearDown() override {
    }

  protected:
    EFIKA_ind_t const nr_  { 4 };
    EFIKA_ind_t const nc_  { 8 };
    EFIKA_ind_t const nnz_ { 8 };
    std::array<EFIKA_ind_t, 5> A_ia_ { 0, 1, 2, 5, 8 };
    std::array<EFIKA_ind_t, 8> A_ja_ { 0, 1, 1, 3, 5, 0, 3, 7 };
    std::array<EFIKA_val_t, 8> A_a_  { 1, 1, 2, 3, 4, 5, 6, 7 };
    std::array<EFIKA_ind_t, 5> B1_ia_ { 0, 1, 2, 5, 8 };
    std::array<EFIKA_ind_t, 8> B1_ja_ { 0, 1, 1, 3, 5, 0, 3, 7 };
    std::array<EFIKA_val_t, 8> B1_a_  { 1, 1, 2, 3, 4, 5, 6, 7 };
    std::array<EFIKA_ind_t, 9> B2_ia_ { 0, 2, 4, 4, 6, 6, 7, 7, 8 };
    std::array<EFIKA_ind_t, 8> B2_ja_ { 0, 3, 1, 2, 2, 3, 2, 3 };
    std::array<EFIKA_val_t, 8> B2_a_  { 1, 5, 1, 2, 3, 6, 4, 7 };
    std::array<EFIKA_ind_t, 5>  C1_ia_;
    std::array<EFIKA_ind_t, 11> C1_ja_;
    std::array<EFIKA_val_t, 11> C1_a_;
    std::array<EFIKA_ind_t, 5>  C2_ia_;
    std::array<EFIKA_ind_t, 11> C2_ja_;
    std::array<EFIKA_val_t, 11> C2_a_;
    std::array<EFIKA_val_t, 8> h_;
    EFIKA_Matrix A_;
    EFIKA_Matrix B1_;
    EFIKA_Matrix B2_;
    EFIKA_Matrix C1_;
    EFIKA_Matrix C2_;
};

} // namespace

TEST_F(BLAS, SpGEMM) {
  efika_BLAS_spgemm_csr_csc(this->A_.nr, this->B1_.nr, this->A_.ia, this->A_.ja,
                            this->A_.a, this->B1_.ia, this->B1_.ja, this->B1_.a,
                            this->C1_.ia, this->C1_.ja, this->C1_.a,
                            this->h_.data());

  efika_BLAS_spgemm_csr_idx(this->A_.nr, this->A_.ia, this->A_.ja, this->A_.a,
                            this->B2_.ia, this->B2_.ja, this->B2_.a,
                            this->C2_.ia, this->C2_.ja, this->C2_.a,
                            this->h_.data());

  ASSERT_EQ(this->C1_.nr, this->C2_.nr);
  ASSERT_EQ(this->C1_.nc, this->C2_.nc);
  ASSERT_EQ(this->C1_.nnz, this->C2_.nnz);
  for (EFIKA_ind_t i = 0; i <= this->C1_.nr; i++)
    ASSERT_EQ(this->C1_.ia[i], this->C2_.ia[i]) << "i = " << i;
}
