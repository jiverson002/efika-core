/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core.h"

#include "efika/core/blas.h"
#include "efika/core/export.h"
#include "efika/core/gc.h"
#include "efika/core/pp.h"
#include "efika/core/rename.h"
#include "efika/core/rsb.h"

/*----------------------------------------------------------------------------*/
/* helper struct for sorting */
/*----------------------------------------------------------------------------*/
struct kv {
  struct coord {
    ind_t r;     /*!< row id */
    ind_t c;     /*!< column id */
  } k;
  val_t v;       /*!< non-zero value */
};

/*----------------------------------------------------------------------------*/
/* helper function for qsort */
/*----------------------------------------------------------------------------*/
static int
kv_cmp_z(void const * const ap, void const * const bp)
{
  struct kv const * const a = (struct kv const*)ap;
  struct kv const * const b = (struct kv const*)bp;

#define less_msb(x, y) ((x) < (y) && (x) < ((x) ^ (y)))

  if (less_msb(a->k.r ^ b->k.r, a->k.c ^ b->k.c))
    return a->k.c < b->k.c ? -1 : 1;
  else
    return a->k.r < b->k.r ? -1 : 1;

#undef less_msb
}

/*----------------------------------------------------------------------------*/
/* helper function for qsort */
/*----------------------------------------------------------------------------*/
static int
kv_cmp_rc(void const * const ap, void const * const bp)
{
  struct kv const * const a = (struct kv const*)ap;
  struct kv const * const b = (struct kv const*)bp;

  return a->k.r < b->k.r ? -1
       : a->k.r > b->k.r ?  1
       : a->k.c < b->k.c ? -1
       : 1;
}

/*----------------------------------------------------------------------------*/
/*! Compress a row and column index into a single ind_t value. The row index
 *  will occupy the upper half and the column index will occupy the lower half.
 */
/*----------------------------------------------------------------------------*/
static inline ind_t
RSB_leaf_zindex(ind_t const ro, ind_t const co, struct coord const c)
{
  return RSB_idx(c.r - ro, c.c - co);
}

/*----------------------------------------------------------------------------*/
/*! Compute the smallest index of an element that is greater than or equal to a
 *  given value. */
/*----------------------------------------------------------------------------*/
static inline ind_t
RSB_node_bsearch(
  int   const row,
  ind_t const k,
  struct kv const * const restrict a,
  ind_t const n
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
RSB_leaf_setup(
  ind_t const ro,
  ind_t const co,
  ind_t const nnz,
  struct kv const * const restrict kv,
  ind_t           * const restrict za,
  val_t           * const restrict a
)
{
  for (ind_t k = 0; k < nnz; k++) {
    za[k] = RSB_leaf_zindex(ro, co, kv[k].k);
    if (a)
      a[k] = kv[k].v;
  }
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static void
RSB_node_split(
  ind_t const ro,
  ind_t const co,
  ind_t const n,
  ind_t const nnz,
  struct kv const * const restrict kv,
  ind_t           * const restrict sa,
  ind_t           * const restrict za,
  val_t           * const restrict a
)
{
  /* don't explicitly split node if dimensions are small enough */
  if (!RSB_is_split(n)) {
    RSB_leaf_setup(ro, co, nnz, kv, za, a);
    return;
  }

  /* compute new dimension */
  ind_t const nn = n / 2;

  /* compute row and column split keys */
  ind_t const rsp = ro + nn;
  ind_t const csp = co + nn;

  /* binary search for quadrant splits */
  sa[1] = RSB_node_bsearch(1, rsp, kv, nnz);
  sa[0] = RSB_node_bsearch(0, csp, kv, sa[1]);
  sa[2] = sa[1] + RSB_node_bsearch(0, csp, kv + sa[1], nnz - sa[1]);

  /* compute size of book-keeping data structure */
  ind_t const nsa = RSB_sa_size(nn);

  /* compute quadrant split offsets */
  ind_t * const sa0 = sa + 3;
  ind_t * const sa1 = sa0 + nsa;
  ind_t * const sa2 = sa1 + nsa;
  ind_t * const sa3 = sa2 + nsa;

  /* recursively split each quadrant */
  RSB_node_split(ro, co, nn, sa[0], kv, sa0, za, a);
  RSB_node_split(ro, co + nn, nn, sa[1] - sa[0], kv + sa[0], sa1, za + sa[0],
                 a ? a + sa[0] : NULL);
  RSB_node_split(ro + nn, co, nn, sa[2] - sa[1], kv + sa[1], sa2, za + sa[1],
                 a ? a + sa[1] : NULL);
  RSB_node_split(ro + nn, co + nn, nn, nnz - sa[2], kv + sa[2], sa3, za + sa[2],
                 a ? a + sa[2] : NULL);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static void
RSB_leaf_walk(
  ind_t const ro,
  ind_t const co,
  ind_t const nnz,
  ind_t     const * const restrict za,
  val_t     const * const restrict a,
  struct kv       * const restrict kv
)
{
  for (ind_t k = 0; k < nnz; k++) {
    /* compensate for compressed leaf indexing */
    kv[k].k.r = ro + RSB_row(za[k]);
    kv[k].k.c = co + RSB_col(za[k]);
    if (a)
      kv[k].v = a[k];
  }
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static void
RSB_node_walk(
  ind_t const ro,
  ind_t const co,
  ind_t const n,
  ind_t const nnz,
  ind_t     const * const restrict sa,
  ind_t     const * const restrict za,
  val_t     const * const restrict a,
  struct kv       * const restrict kv
)
{
  /* don't explicitly split node if dimensions are small enough */
  if (!RSB_is_split(n)) {
    RSB_leaf_walk(ro, co, nnz, za, a, kv);
    return;
  }

  /* compute new dimension */
  ind_t const nn = n / 2;

  /* compute quadrant # non-zeros */
  ind_t const nnz0 = sa[0];
  ind_t const nnz1 = sa[1] - sa[0];
  ind_t const nnz2 = sa[2] - sa[1];
  ind_t const nnz3 = nnz   - sa[2];

  /* compute number of splits per quadrant */
  ind_t const nsa = RSB_sa_size(nn);

  /* compute quadrant split offsets */
  ind_t const * const sa0 = sa + 3;
  ind_t const * const sa1 = sa0 + nsa;
  ind_t const * const sa2 = sa1 + nsa;
  ind_t const * const sa3 = sa2 + nsa;

  /* recursively walk each quadrant */
  RSB_node_walk(ro, co, nn, nnz0, sa0, za, a, kv);
  RSB_node_walk(ro, co + nn, nn, nnz1, sa1, za + sa[0], a ? a + sa[0] : NULL,
                kv + sa[0]);
  RSB_node_walk(ro + nn, co, nn, nnz2, sa2, za + sa[1], a ? a + sa[1] : NULL,
                kv + sa[1]);
  RSB_node_walk(ro + nn, co + nn, nn, nnz3, sa3, za + sa[2],
                a ? a + sa[2] : NULL, kv + sa[2]);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static int
csrcsc(Matrix const * const A, Matrix * const B)
{
  /* ...garbage collected function... */
  GC_func_init();

  /* unpack /A/ */
  ind_t const a_nr  = A->nr;
  ind_t const a_nc  = A->nc;
  ind_t const a_nnz = A->nnz;
  ind_t const * const a_ia = A->ia;
  ind_t const * const a_ja = A->ja;
  val_t const * const a_a  = A->a;

  if (!pp_all(a_ia, a_ja))
    return -1;

  /* allocate memory for inverted index */
  ind_t * const b_ia = GC_malloc((a_nc + 1) * sizeof(*b_ia));
  ind_t * const b_ja = GC_malloc(a_nnz * sizeof(*b_ja));
  val_t * const b_a  = GC_malloc(a_nnz * sizeof(*b_a));

  BLAS_csrcsc(a_nr, a_nc, a_ia, a_ja, a_a, b_ia, b_ja, b_a);

  /* record relevant info in /B/ */
  B->mord = MORD_CSC;
  B->sort = NONE;
  B->nr   = a_nc;
  B->nc   = a_nr;
  B->nnz  = a_nnz;
  B->ia   = b_ia;
  B->ja   = b_ja;
  B->a    = b_a;

  return 0;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static int
csccsr(Matrix const * const A, Matrix * const B)
{
  return csrcsc(A, B);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static int
csrrsb(Matrix const * const A, Matrix * const B)
{
  /* ...garbage collected function... */
  GC_func_init();

  /* unpack /A/ */
  ind_t const a_nr  = A->nr;
  ind_t const a_nc  = A->nc;
  ind_t const a_nnz = A->nnz;
  ind_t const * const a_ia = A->ia;
  ind_t const * const a_ja = A->ja;
  val_t const * const a_a  = A->a;

  if (!pp_all(a_ia, a_ja))
    return -1;

  /* allocate temporary storage */
  struct kv * const kv = GC_malloc(a_nnz * sizeof(*kv));

  /* populate key-value of each non-zero */
  for (ind_t i = 0, k = 0; i < a_nr; i++) {
    for (ind_t j = a_ia[i]; j < a_ia[i + 1]; j++, k++) {
      kv[k].k.r = i;
      kv[k].k.c = a_ja[j];
      if (a_a)
        kv[k].v = a_a[j];
    }
  }

  /* sort non-zeros by z-index */
  qsort(kv, a_nnz, sizeof(*kv), kv_cmp_z);

  /* dimensions need to be powers of two for proper binary searches */
  ind_t const n = RSB_size(a_nr, a_nc);

  /* allocate new storage */
  ind_t * const b_sa = GC_malloc(RSB_sa_size(n) * sizeof(*b_sa));
  ind_t * const b_za = GC_malloc(a_nnz * sizeof(*b_za));
  val_t * b_a = NULL;
  if (a_a)
    b_a = GC_malloc(a_nnz * sizeof(*b_a));

  /* setup book-keeping */
  RSB_node_split(0, 0, n, a_nnz, kv, b_sa, b_za, b_a);

  /* record relevant info in /B/ */
  B->sort = NONE;
  B->mord = MORD_RSB;
  B->nr   = a_nr;
  B->nc   = a_nc;
  B->nnz  = a_nnz;
  B->sa   = b_sa;
  B->za   = b_za;
  B->a    = b_a;

  /* free temporary storage */
  GC_free(kv);

  return 0;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static int
rsbcsr(Matrix const * const A, Matrix * const B)
{
  /* ...garbage collected function... */
  GC_func_init();

  /* unpack /A/ */
  ind_t const a_nr  = A->nr;
  ind_t const a_nc  = A->nc;
  ind_t const a_nnz = A->nnz;
  ind_t const * const a_sa = A->sa;
  ind_t const * const a_za = A->za;
  val_t const * const a_a  = A->a;

  if (!a_za)
    return -1;

  /* allocate temporary storage */
  struct kv * const kv = GC_malloc(a_nnz * sizeof(*kv));

  /* dimensions need to be powers of two for proper binary searches */
  ind_t const n = RSB_size(a_nr, a_nc);

  /* populate key-value of each non-zero */
  RSB_node_walk(0, 0, n, a_nnz, a_sa, a_za, a_a, kv);

  /* sort non-zeros by row then column */
  qsort(kv, a_nnz, sizeof(*kv), kv_cmp_rc);

  /* allocate new storage */
  ind_t * const b_ia = GC_malloc((a_nr + 1) * sizeof(*b_ia));
  ind_t * const b_ja = GC_malloc(a_nnz * sizeof(*b_ja));
  val_t * b_a = NULL;
  if (a_a)
    b_a = GC_malloc(a_nnz * sizeof(*b_a));

  /* copy memory */
  b_ia[0] = 0;
  for (ind_t i = 0, k = 0; i < a_nnz;) {
    ind_t const r = kv[i].k.r;

    for (; kv[i].k.r == r; i++) {
      if (a_a)
        b_a[k]  = kv[i].v;
      b_ja[k++] = kv[i].k.c;
    }

    b_ia[r + 1] = k;
  }

  /* record relevant info in /B/ */
  B->sort = NONE;
  B->mord = MORD_CSR;
  B->nr   = a_nr;
  B->nc   = a_nc;
  B->nnz  = a_nnz;
  B->ia   = b_ia;
  B->ja   = b_ja;
  B->a    = b_a;

  /* free temporary storage */
  GC_free(kv);

  return 0;
}

/*----------------------------------------------------------------------------*/
/*! Function to convert a matrix from one storage format to another. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT int
Matrix_conv(Matrix const * const A, Matrix * const B, int const which)
{
  if (!pp_all(A, B))
    return -1;

  if (A->mord == which)
    return 0;

#define combine(a, b) ((a) << 8 | (b))

  switch (combine(A->mord, which)) {
    case combine(MORD_CSR, MORD_CSC):
    return csrcsc(A, B);

    case combine(MORD_CSC, MORD_CSR):
    return csccsr(A, B);

    case combine(MORD_CSR, MORD_RSB):
    return csrrsb(A, B);

    case combine(MORD_RSB, MORD_CSR):
    return rsbcsr(A, B);

    default:
    return -1;
  }

#undef combine
}
