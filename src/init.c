/* SPDX-License-Identifier: MIT */
#include <string.h>

#include "efika/core/export.h"
#include "efika/core/rename.h"
#include "efika/core.h"

/*----------------------------------------------------------------------------*/
/*! Initialize matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT int
Matrix_init(Matrix * const M)
{
  memset(M, 0, sizeof(*M));

  return 0;
}
