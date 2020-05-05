/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core/rename.h"
#include "efika/core.h"

/*----------------------------------------------------------------------------*/
/*! Free resources of matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_EXPORT void
Matrix_free(Matrix * const M)
{
  free(M->ia);
  free(M->ja);
  free(M->a);
  free(M->vsiz);
  free(M->vwgt);

  free(M->sa);
  free(M->za);

  if (M->pp) {
    M->pp_free(M->pp);
    free(M->pp);
  }
}
