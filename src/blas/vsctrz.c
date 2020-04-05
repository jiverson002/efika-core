/* SPDX-License-Identifier: MIT */
#include "efika/core/blas.h"

/*----------------------------------------------------------------------------*/
/*! Zeros the corresponding values of a compressed sparse vector in full storage
 * form. */
/*----------------------------------------------------------------------------*/
void
BLAS_vsctrz(
  ind_t const nz,
  ind_t const * const restrict indx,
  val_t       * const restrict y
)
{
  for (ind_t i = 0; i < nz; i++)
    y[indx[i]] = 0.0;
}
