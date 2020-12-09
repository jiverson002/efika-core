/* SPDX-License-Identifier: MIT */
#include "efika/core.h"

#include "efika/core/blas.h"

/*----------------------------------------------------------------------------*/
/*! Dot product of a sparse vector x in compressed-vector storage mode and a
 *  sparse vector y in full-vector storage mode. */
/*----------------------------------------------------------------------------*/
val_t
BLAS_vdoti(
  ind_t const nz,
  val_t const * const restrict x,
  ind_t const * const restrict indx,
  val_t const * const restrict y
)
{
  val_t res = 0.0;
  for (ind_t i = 0; i < nz; i++)
    res += x[i] * y[indx[i]];
  return res;
}
