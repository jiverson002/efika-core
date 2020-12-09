/* SPDX-License-Identifier: MIT */
#include <string.h>

#include "efika/core.h"

#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/*! Initialize matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_EXPORT int
Matrix_init(Matrix * const M)
{
  memset(M, 0, sizeof(*M));

  return 0;
}
