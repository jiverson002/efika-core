/* SPDX-License-Identifier: MIT */
#include "efika/core/blas.h"

/*----------------------------------------------------------------------------*/
/*! Converts compressed sparse vectors into full storage form. */
/*----------------------------------------------------------------------------*/
void
BLAS_vgthrz(
  ind_t const nz,
  val_t       * const restrict y,
  val_t       * const restrict x,
  ind_t const * const restrict indx
)
{
  for (ind_t i = 0; i < nz; i++) {
    x[i] = y[indx[i]];
    y[indx[i]] = 0.0;
  }
}
