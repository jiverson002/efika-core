/* SPDX-License-Identifier: MIT */
#include <assert.h>
#include <limits.h>
#include <stdlib.h>

#include <stdio.h>

#include "efika/core.h"

#include "efika/core/export.h"
#include "efika/core/gc.h"
#include "efika/core/pp.h"
#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/* helper struct for sorting */
/*----------------------------------------------------------------------------*/
struct coord {
  ind_t r; /*!< row id */
  ind_t c; /*!< column id */
};

struct zindex {
  ind_t u; /*!< upper half of number */
  ind_t l; /*!< lower half of number */
};

struct kv {
  struct zindex k; /*!< z-index of non-zero value */
  val_t v;         /*!< non-zero value */
};

/*----------------------------------------------------------------------------*/
/* helper functions to handle a z-indices */
/*----------------------------------------------------------------------------*/
static inline ind_t
to_zindex_half(ind_t const r, ind_t const c)
{
  ind_t const one = 1;
  ind_t z = 0;

  for (size_t i = 0; i < sizeof(ind_t) * CHAR_BIT / 2; i++)
    z |= (r & one << i) << (i + 1) | (c & one << i) << i;

  return z;
}

static inline struct coord
from_zindex_half(ind_t const z)
{
  ind_t const one = 1;
  struct coord rc = { 0, 0 };

  for (size_t i = 0; i < sizeof(ind_t) * CHAR_BIT / 2; i++) {
    rc.r |= (z & one << i << (i + 1)) >> (i + 1);
    rc.c |= (z & one << i << (i + 0)) >> (i + 0);
  }

  return rc;
}

static inline struct zindex
to_zindex(ind_t const r, ind_t const c)
{
  size_t const half = sizeof(ind_t) * CHAR_BIT / 2;
  struct zindex z = { 0, 0 };

  z.l = to_zindex_half(r, c);
  z.u = to_zindex_half(r >> half, c >> half);

  return z;
}

static inline ind_t
from_zindex(struct zindex const z)
{
  size_t const half = sizeof(ind_t) * CHAR_BIT / 2;
  struct coord res;
  struct coord rc = { 0, 0 };

  res = from_zindex_half(z.l);
  rc.r |= res.r;
  rc.c |= res.c;
  res = from_zindex_half(z.u);
  rc.r |= res.r << half;
  rc.c |= res.c << half;

  assert(rc.r < 65536);
  assert(rc.c < 65536);

  return rc.r << half | rc.c;
}

/*----------------------------------------------------------------------------*/
/* helper function for qsort */
/*----------------------------------------------------------------------------*/
static int
kv_cmp(void const * const ap, void const * const bp)
{
  struct kv const * const a = (struct kv const*)ap;
  struct kv const * const b = (struct kv const*)bp;

  return a->k.u < b->k.u ? -1 : a->k.u > b->k.u ? 1 : a->k.l < b->k.l ? -1 : 1;
}

/*----------------------------------------------------------------------------*/
/*! Function to re-order rows of a matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT int
Matrix_zord(Matrix const * const M, Matrix * const Z)
{
  /* ...garbage collected function... */
  GC_func_init();

  if (!pp_all(M, Z))
    return -1;

  /* unpack /M/ */
  ind_t const nr  = M->nr;
  ind_t const nnz = M->nnz;
  ind_t const * const m_ia = M->ia;
  ind_t const * const m_ja = M->ja;
  val_t const * const m_a  = M->a;

  if (!pp_all(m_ia, m_ja))
    return -1;

  assert(nr < 65536);
  assert(M->nc < 65536);

  /* allocate temporary storage */
  struct kv * const kv = GC_malloc(nnz * sizeof(*kv));

  /* compute z-indices of each non-zero */
  for (ind_t i = 0, k = 0; i < nr; i++) {
    for (ind_t j = m_ia[i]; j < m_ia[i + 1]; j++, k++) {
      kv[k].k = to_zindex(i, m_ja[j]);
      kv[k].v = m_a[j];
    }
  }

  /* sort non-zeros by z-index */
  qsort(kv, nnz, sizeof(*kv), kv_cmp);

  /* allocate new storage */
  ind_t * const z_za = GC_malloc(nnz * sizeof(*z_za));
  val_t * z_a = NULL;
  if (m_a)
    z_a = GC_malloc(nnz * sizeof(*z_a));

  /* setup z-ordered matrix */
  for (ind_t k = 0; k < nnz; k++) {
    z_za[k] = from_zindex(kv[k].k);
    if (m_a)
      z_a[k] = kv[k].v;
  }

  /* record values in /Z/ */
  Z->nr  = nr;
  Z->nc  = M->nc;
  Z->nnz = nnz;
  Z->za  = z_za;
  Z->a   = z_a;

  /* free temporary storage */
  GC_free(kv);

  return 0;
}
