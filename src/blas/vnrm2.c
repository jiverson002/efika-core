/* SPDX-License-Identifier: MIT */
#include <math.h>

#include "efika/core.h"

#include "efika/core/blas.h"

/*----------------------------------------------------------------------------*/
/*! Compute the Euclidean length (l_2 norm) of vector x. */
/*----------------------------------------------------------------------------*/
val_t
BLAS_vnrm2(
  ind_t const n,
  val_t const * const restrict x
)
{
  val_t res = 0.0;
  for (ind_t i = 0; i < n; i++)
    res += x[i] * x[i];
  return sqrtv(res);
}
