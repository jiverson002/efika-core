/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core/export.h"
#include "efika/core/rename.h"
#include "efika/core.h"

/*----------------------------------------------------------------------------*/
/*! Free resources of matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT void
Matrix_free(Matrix * const M)
{
  free(M->ia);
  free(M->ja);
  free(M->a);
  free(M->vsiz);
  free(M->vwgt);
}