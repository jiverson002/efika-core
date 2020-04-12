/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core.h"

#include "efika/core/export.h"
#include "efika/core/gc.h"
#include "efika/core/pp.h"
#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/* helper structs for sorting                                                 */
/*----------------------------------------------------------------------------*/
struct kv
{
  union {
    ind_t i;
    val_t f;
  } k;
  ind_t v;
};

/*----------------------------------------------------------------------------*/
/* helper functions for qsort                                                 */
/*----------------------------------------------------------------------------*/
static int
kvi_asc(void const * const a, void const * const b)
{
  return (((struct kv const *)a)->k.i < ((struct kv const *)b)->k.i) ? -1 : 1;
}

static int
kvi_dsc(void const * const a, void const * const b)
{
  return (((struct kv const *)a)->k.i > ((struct kv const *)b)->k.i) ? -1 : 1;
}

#if 0
static int
kvf_asc(void const * const a, void const * const b)
{
  return (((struct kv const *)a)->k.f < ((struct kv const *)b)->k.f) ? -1 : 1;
}

static int
kvf_dsc(void const * const a, void const * const b)
{
  return (((struct kv const *)a)->k.f > ((struct kv const *)b)->k.f) ? -1 : 1;
}
#endif

/*----------------------------------------------------------------------------*/
/*! Function to degree re-order columns of a matrix. */
/*----------------------------------------------------------------------------*/
static int
cord_deg(Matrix * const M, int const order)
{
  /* ...garbage collected function... */
  GC_func_init();

  /* unpack /M/ */
  ind_t const nc  = M->nc;
  ind_t const nnz = M->nnz;
  ind_t       * const ja  = M->ja;

  /* validate input */
  if (ASC != order && DSC != order)
    return -1;
  if (!pp_all(nc, nnz, ja))
    return -1;

  /* allocate scratch memory */
  struct kv * const kv  = GC_malloc(nc * sizeof(*kv));
  ind_t     * const map = GC_malloc(nc * sizeof(*map));

  for (ind_t i = 0; i < nc; i++) {
    kv[i].k.i = 0;
    kv[i].v   = i;
  }

  for (ind_t i = 0; i < nnz; i++)
    kv[ja[i]].k.i++;

  qsort(kv, nc, sizeof(*kv), ASC == order ? kvi_asc : kvi_dsc);

  for (ind_t i = 0; i < nc; i++)
    map[kv[i].v] = i;

  for (ind_t i = 0; i < nnz; i++)
    ja[i] = map[ja[i]];

  /* record relevant info in /M/ */
  M->sort = NONE;
  M->symm = 0;

  /* free scratch memory */
  GC_free(kv);
  GC_free(map);

  return 0;
}

/*----------------------------------------------------------------------------*/
/*! Function to re-order columns of a matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT int
Matrix_cord(Matrix * const m, int const flags)
{
  switch (flags & TYPE_FLAGS) {
    case DEG:
      return cord_deg(m, flags & ORDER_FLAGS);
    default:
      return -1;
  }
}
