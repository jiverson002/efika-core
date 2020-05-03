/* SPDX-License-Identifier: MIT */
#include "efika/core/blas.h"

/*----------------------------------------------------------------------------*/
/*! Sparse-sparse matrix multiplication C = A * B, where A is stored in CSR
 * format and B is stored in CSC format.
 *
 *  Algorithm:
 *    for each row i of A:
 *      load row i's values into a hash table (dense array)
 *      for each column of B:
 *        compute sparse dot product with hash table
 *        record result in C
 *      clear hash table
 */
/*----------------------------------------------------------------------------*/
void
BLAS_spgemm_csr_csc(
  ind_t const m,
  ind_t const p,
  ind_t const * const restrict ia,
  ind_t const * const restrict ja,
  val_t const * const restrict a,
  ind_t const * const restrict ib,
  ind_t const * const restrict jb,
  val_t const * const restrict b,
  ind_t       * const restrict ic,
  ind_t       * const restrict jc,
  val_t       * const restrict c,
  val_t       * const restrict h
)
{
  ic[0] = 0;
  for (ind_t i = 0, nnz = 0; i < m; i++) {
    BLAS_vsctr(ia[i + 1] - ia[i], a + ia[i], ja + ia[i], h);

    for (ind_t j = 0; j < p; j++) {
      c[nnz] = BLAS_vdoti(ib[j + 1] - ib[j], b + ib[j], jb + ib[j], h);
      if (c[nnz] > 0.0)
        jc[nnz++] = j;
    }

    BLAS_vsctrz(ia[i + 1] - ia[i], ja + ia[i], h);

    ic[i + 1] = nnz;
  }
}

/*----------------------------------------------------------------------------*/
/*! Sparse-sparse matrix multiplication C = A * B, where A and B are stored in
 *  CSR format and B is treated as an inverted index.
 *
 *  Algorithm:
 *    for each row i in A:
 *      for each non-zero column x with value v:
 *        for each row y in B that has a non-zero in column x with value w:
 *          accumulate the multiplication v * w in hash table (dense array) for
 *          key y
 *      gather results from hash table into C
 *      clear hash table
 */
/*----------------------------------------------------------------------------*/
void
BLAS_spgemm_csr_idx(
  ind_t const m,
  ind_t const * const restrict ia,
  ind_t const * const restrict ja,
  val_t const * const restrict a,
  ind_t const * const restrict ib,
  ind_t const * const restrict jb,
  val_t const * const restrict b,
  ind_t       * const restrict ic,
  ind_t       * const restrict jc,
  val_t       * const restrict c,
  val_t       * const restrict spa
)
{
  ic[0] = 0;
  for (ind_t i = 0, nnz = 0; i < m; i++) {
    for (ind_t j = ia[i]; j < ia[i + 1]; j++) {
      ind_t const x = ja[j];
      val_t const v = a[j];

      for (ind_t k = ib[x]; k < ib[x + 1]; k++) {
        ind_t const y = jb[k];
        val_t const w = b[k];

        if (0.0 == spa[y])
          jc[nnz++] = y;

        spa[y] += v * w;

        /* correct floating-point underflow, when v * w == 0.0 */
        nnz -= (0.0 == spa[y]);
      }
    }

    ic[i + 1] = nnz;

    BLAS_vgthrz(ic[i + 1] - ic[i], spa, c + ic[i], jc + ic[i]);
  }
}

/*----------------------------------------------------------------------------*/
/*! Sparse-sparse matrix multiplication B = A * A', where A is stored in RSB
 *  format and A' is not stored explicitly.
 */
/*----------------------------------------------------------------------------*/
void
BLAS_spgemm_rsb_rsb(
  ind_t const nnz,
  ind_t const * const restrict za,
  val_t const * const restrict a,
  ind_t const * const restrict zb,
  val_t const * const restrict b,
  ind_t       * const restrict zc,
  val_t       * const restrict c
)
{
  for (ind_t i = 0, nnnz = 0; i < nnz; i++) {
    (void)nnnz;
  }
  (void)za;
  (void)a;
  (void)zb;
  (void)b;
  (void)zc;
  (void)c;
}
