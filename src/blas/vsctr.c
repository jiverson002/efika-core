/* SPDX-License-Identifier: MIT */
#include "efika/core/blas.h"

/*----------------------------------------------------------------------------*/
/*! Converts compressed sparse vectors into full storage form. */
/*----------------------------------------------------------------------------*/
void
BLAS_vsctr(
  ind_t const nz,
  val_t const * const restrict x,
  ind_t const * const restrict indx,
  val_t       * const restrict y
)
{
  for (ind_t i = 0; i < nz; i++)
    y[indx[i]] = x[i];
}
