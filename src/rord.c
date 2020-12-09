/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core.h"

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

/*----------------------------------------------------------------------------*/
/*! Function to degree re-order rows of a matrix. */
/*----------------------------------------------------------------------------*/
static int
rord_deg(Matrix * const M, int const order)
{
  /* ...garbage collected function... */
  GC_func_init();

  if (ASC != order && DSC != order)
    return -1;

  ind_t const nr = M->nr;
  ind_t const * const ia = M->ia;

  ind_t     * const iperm = GC_malloc(nr * sizeof(*iperm));
  struct kv * const kv    = GC_malloc(nr * sizeof(*kv));

  for (ind_t i = 0; i < nr; i++) {
    kv[i].k.i = ia[i + 1] - ia[i];
    kv[i].v   = i;
  }

  qsort(kv, nr, sizeof(*kv), ASC == order ? kvi_asc : kvi_dsc);

  for (ind_t i = 0; i < nr; i++)
    iperm[i] = kv[i].v;

  int err = Matrix_perm(M, NULL, iperm);
  GC_assert(!err);

  GC_free(iperm);
  GC_free(kv);

  return 0;
}

/*----------------------------------------------------------------------------*/
/*! Function to value re-order rows of a matrix. */
/*----------------------------------------------------------------------------*/
static int
rord_val(Matrix * const M, int const order)
{
  /* ...garbage collected function... */
  GC_func_init();

  if (ASC != order && DSC != order)
    return -1;
  if (!M->a)
    return -1;

  ind_t const nr = M->nr;
  ind_t const * const ia = M->ia;
  val_t const * const a  = M->a;

  ind_t     * const iperm = GC_malloc(nr * sizeof(*iperm));
  struct kv * const kv    = GC_malloc(nr * sizeof(*kv));

  for (ind_t i = 0; i < nr; i++) {
    kv[i].k.f = VAL_MIN;
    kv[i].v   = i;
    for (ind_t j = ia[i]; j < ia[i + 1]; j++)
      if (a[j] > kv[i].k.f)
        kv[i].k.f = a[j];
  }

  qsort(kv, nr, sizeof(*kv), ASC == order ? kvf_asc : kvf_dsc);

  for (ind_t i = 0; i < nr; i++)
    iperm[i] = kv[i].v;

  int err = Matrix_perm(M, NULL, iperm);
  GC_assert(!err);

  GC_free(iperm);
  GC_free(kv);

  return 0;
}

/*----------------------------------------------------------------------------*/
/*! Function to re-order rows of a matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_EXPORT int
Matrix_rord(Matrix * const M, int const flags)
{
  switch (flags & TYPE_FLAGS) {
    case DEG:
      return rord_deg(M, flags & ORDER_FLAGS);
    case VAL:
      return rord_val(M, flags & ORDER_FLAGS);
    default:
      return -1;
  }
}
