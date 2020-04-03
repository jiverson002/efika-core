/* SPDX-License-Identifier: MIT */
#include <string.h>

#include "efika/core/blas.h"

/*----------------------------------------------------------------------------*/
/*! Converts a sparse matrix in the CSR format to the CSC format. */
/*----------------------------------------------------------------------------*/
void
BLAS_csrcsc(
  ind_t const m,
  ind_t const n,
  ind_t const * const restrict ia,
  ind_t const * const restrict ja,
  val_t const * const restrict acsr,
  ind_t       * const restrict ia1,
  ind_t       * const restrict ja1,
  val_t       * const restrict acsc
)
{
  memset(ia1, 0, (n + 1) * sizeof(*ia1));

  for (ind_t i = 0; i < m; i++)
    for (ind_t j = ia[i]; j < ia[i + 1]; j++)
      ia1[ja[j]]++;

  for (ind_t i = 0, p = 0; i <= n; i++) {
    ind_t const t = ia1[i];
    ia1[i] = p;
    p += t;
  }

  for (ind_t i = 0; i < m; i++) {
    for (ind_t j = ia[i]; j < ia[i + 1]; j++) {
      ja1[ia1[ja[j]]]    = i;
      acsc[ia1[ja[j]]++] = acsr[j];
    }
  }

  for (ind_t i = n; i > 0; i--)
    ia1[i] = ia1[i - 1];
  ia1[0] = 0;
}
