/* SPDX-License-Identifier: MIT */
#include "efika/core.h"

#include "efika/core/blas.h"
#include "efika/core/export.h"

/*----------------------------------------------------------------------------*/
/*! Multiply a vector x by a scalar and store in the vector x. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT void
BLAS_vscal(
  ind_t const n,
  val_t const a,
  val_t * const restrict x
)
{
  for (ind_t i = 0; i < n; i++)
    x[i] *= a;
}
