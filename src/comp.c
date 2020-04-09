/* SPDX-License-Identifier: MIT */
#include "efika/core.h"

#include "efika/core/export.h"
#include "efika/core/gc.h"
#include "efika/core/pp.h"
#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/*! Compacts the column-space of the matrix by removing empty columns. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT int
Matrix_comp(Matrix * const M)
{
  /* ...garbage collected function... */
  GC_func_init();

  /* unpack /M/ */
  ind_t const         nc  = M->nc;
  ind_t const         nnz = M->nnz;
  ind_t       * const ja  = M->ja;

  /* validate input */
  if (!pp_all(nc, nnz, ja))
    return -1;

  /* allocate scratch memory */
  ind_t * const map = GC_calloc(nc, sizeof(*map));

  /* mark non-empty columns */
  for (ind_t i = 0; i < nnz; i++)
    map[ja[i]] = 1;

  /* create a mapping from old column id to new column id */
  ind_t nnc = 0;
  for (ind_t i = 0; i < nc; i++)
    if (0 != map[i])
      map[i] = nnc++;

  /* updated ja array with new columns ids */
  for (ind_t i = 0; i < nnz; i++)
    ja[i] = map[ja[i]];

  /* record relevant info in /M/ */
  M->sort = NONE;
  M->symm = 0;
  M->nc   = nnc;

  /* free scratch memory */
  GC_free(map);

  return 0;
}
