/* SPDX-License-Identifier: MIT */
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>

#include <assert.h>
#include <stdio.h>

#include "efika/core.h"

#include "efika/core/export.h"
#include "efika/core/gc.h"
#include "efika/core/pp.h"
#include "efika/core/rename.h"

#define RSB_MIN_NODE_SIZE 1024

/*----------------------------------------------------------------------------*/
/* helper struct for sorting */
/*----------------------------------------------------------------------------*/
struct zindex {
  ind_t r; /*!< row id */
  ind_t c; /*!< column id */
};

struct kv {
  struct zindex k; /*!< z-index of non-zero value */
  val_t v;         /*!< non-zero value */
};

/*----------------------------------------------------------------------------*/
/* helper functions for qsort */
/*----------------------------------------------------------------------------*/
static inline bool less_msb(ind_t const x, ind_t const y)
{
  return x < y && x < (x ^ y);
}

static int
kv_cmp(void const * const ap, void const * const bp)
{
  struct kv const * const a = (struct kv const*)ap;
  struct kv const * const b = (struct kv const*)bp;

  if (less_msb(a->k.r ^ b->k.r, a->k.c ^ b->k.c))
    return a->k.c < b->k.c ? -1 : 1;
  else
    return a->k.r < b->k.r ? -1 : 1;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline ind_t
rsb_block_zindex(ind_t const ro, ind_t const co, struct zindex const z)
{
  size_t const half = sizeof(ind_t) * CHAR_BIT / 2;
  return (z.r - ro) << half | (z.c - co);
}

/*----------------------------------------------------------------------------*/
/*! Compute the smallest index of an element that is greater than or equal to a
 *  given value. */
/*----------------------------------------------------------------------------*/
static inline ind_t
rsb_bsearch(
  int       const row,
  ind_t     const k,
  struct kv const * const restrict a,
  ind_t     const n
)
{
  ind_t l = 0, r = n;

  while (l < r) {
    ind_t const m = l + (r - l) / 2;
    if ((row ? a[m].k.r : a[m].k.c) < k)
      l = m + 1;
    else
      r = m;
  }

  return l;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline void
rsb_setup_leaf(
  ind_t     const ro,
  ind_t     const co,
  ind_t     const nnz,
  struct kv const * const restrict kv,
  ind_t           * const restrict za,
  val_t           * const restrict a
)
{
  for (ind_t k = 0; k < nnz; k++) {
    za[k] = rsb_block_zindex(ro, co, kv[k].k);
    if (a)
      a[k] = kv[k].v;
  }
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static ind_t
rsb_setup_node(
  ind_t     const ro,
  ind_t     const co,
  ind_t     const nr,
  ind_t     const nc,
  ind_t     const nnz,
  struct kv const * const restrict kv,
  ind_t           * const restrict sa,
  ind_t           * const restrict za,
  val_t           * const restrict a
)
{
  /* don't split node if it is too small */
  if (nnz < RSB_MIN_NODE_SIZE)
    return rsb_setup_leaf(ro, co, nnz, kv, za, a), 0;

  /* compute row and column split keys */
  ind_t const rsp = ro + nr / 2;
  ind_t const csp = co + nc / 2;

  /* binary search for quadrant splits */
  sa[1] = rsb_bsearch(1, rsp, kv, nnz);
  sa[0] = rsb_bsearch(0, csp, kv, sa[1]);
  sa[2] = sa[1] + rsb_bsearch(0, csp, kv + sa[1], nnz - sa[1]);

  /* sanity check */
  for (ind_t i = 0; i < sa[0]; i++) {
    assert(kv[i].k.r <  rsp);
    assert(kv[i].k.c <  csp);
  }
  for (ind_t i = sa[0]; i < sa[1]; i++) {
    assert(kv[i].k.r <  rsp);
    assert(kv[i].k.c >= csp);
  }
  for (ind_t i = sa[1]; i < sa[2]; i++) {
    assert(kv[i].k.r >= rsp);
    assert(kv[i].k.c <  csp);
  }
  for (ind_t i = sa[2]; i < nnz; i++) {
    assert(kv[i].k.r >= rsp);
    assert(kv[i].k.c >= csp);
  }

  /* record interval / leaf status for quadrants */
  sa[3] = (sa[0]         > RSB_MIN_NODE_SIZE)
        | (sa[1] - sa[0] > RSB_MIN_NODE_SIZE) << 1
        | (sa[2] - sa[1] > RSB_MIN_NODE_SIZE) << 2
        | (nnz   - sa[2] > RSB_MIN_NODE_SIZE) << 3;

  /* compute quadrant dimensions */
  ind_t const nrt = nr / 2;
  ind_t const nrb = nr - nrt;
  ind_t const ncl = nc / 2;
  ind_t const ncr = nc - ncl;

  /* compute quadrant offsets */
  ind_t const ob = ro + nrt;
  ind_t const or = co + ncl;

  /* */
  ind_t nsa = 3;

  /* recursively setup each quadrant */
  nsa += rsb_setup_node(ro, co, nrt, ncl, sa[0], kv, sa + nsa, za, a);
  nsa += rsb_setup_node(ro, or, nrt, ncr, sa[1] - sa[0], kv + sa[0], sa + nsa,
                        za + sa[0], a ? a + sa[0] : NULL);
  nsa += rsb_setup_node(ob, co, nrb, ncl, sa[2] - sa[1], kv + sa[1], sa + nsa,
                        za + sa[1], a ? a + sa[1] : NULL);
  nsa += rsb_setup_node(ob, or, nrb, ncr, nnz - sa[2], kv + sa[2], sa + nsa,
                        za + sa[2], a ? a + sa[2] : NULL);

  return nsa;
}

/*----------------------------------------------------------------------------*/
/*! Function to re-order rows of a matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT int
Matrix_rsb(Matrix const * const M, Matrix * const Z)
{
  /* ...garbage collected function... */
  GC_func_init();

  if (!pp_all(M, Z))
    return -1;

  /* unpack /M/ */
  ind_t const nr  = M->nr;
  ind_t const nc  = M->nc;
  ind_t const nnz = M->nnz;
  ind_t const * const m_ia = M->ia;
  ind_t const * const m_ja = M->ja;
  val_t const * const m_a  = M->a;

  if (!pp_all(m_ia, m_ja))
    return -1;

  /* allocate temporary storage */
  struct kv * const kv = GC_malloc(nnz * sizeof(*kv));

  /* populate key-value of each non-zero */
  for (ind_t i = 0, k = 0; i < nr; i++) {
    for (ind_t j = m_ia[i]; j < m_ia[i + 1]; j++, k++) {
      kv[k].k.r = i;
      kv[k].k.c = m_ja[j];
      kv[k].v   = m_a[j];
    }
  }

  /* sort non-zeros by z-index */
  qsort(kv, nnz, sizeof(*kv), kv_cmp);

  /* allocate new storage */
  ind_t * const z_sa = GC_malloc(4 * nnz * sizeof(*z_sa));
  ind_t * const z_za = GC_malloc(nnz * sizeof(*z_za));
  val_t * z_a = NULL;
  if (m_a)
    z_a = GC_malloc(nnz * sizeof(*z_a));

  // XXX: Dimensions need to be powers of two for proper binary searches
  /* setup book-keeping */
  //ind_t const nsa = rsb_setup_node(0, 0, nr, nc, nnz, kv, z_sa, z_za, z_a);
  ind_t const nsa = rsb_setup_node(0, 0, 65536, 65536, nnz, kv, z_sa, z_za, z_a);
  (void)nc;

  /* record values in /Z/ */
  Z->nr  = nr;
  Z->nc  = M->nc;
  Z->nnz = nnz;
  Z->sa  = GC_realloc(z_sa, nsa * sizeof(*z_sa));
  Z->za  = z_za;
  Z->a   = z_a;

  /* free temporary storage */
  GC_free(kv);

  return 0;
}
