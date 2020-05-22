/* SPDX-License-Identifier: MIT */
#ifndef EFIKA_CORE_RSB_H
#define EFIKA_CORE_RSB_H 1

#include <limits.h>
#include <stdbool.h>
#include <string.h>

#include <stdio.h>

#include "efika/core.h"

#include "efika/core/rename.h"

#ifdef __cplusplus
# ifndef restrict
#   define undef_restrict
#   define restrict
# endif
#endif

/*----------------------------------------------------------------------------*/
/*! RSB routines. */
/*----------------------------------------------------------------------------*/
#define RSB_spgemm efika_RSB_spgemm

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static ind_t const RSB_SHIFT = sizeof(ind_t) * CHAR_BIT / 2;

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static ind_t const RSB_SIZE = (ind_t)1 << RSB_SHIFT;

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static ind_t const RSB_MASK = RSB_SIZE - 1;

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline ind_t
RSB_row(ind_t const x)
{
  return x >> RSB_SHIFT;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline ind_t
RSB_col(ind_t const x)
{
  return x & RSB_MASK;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline ind_t
RSB_idx(ind_t const r, ind_t const c)
{
  return r << RSB_SHIFT | c;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline bool
RSB_is_split(ind_t const n)
{
  return n > RSB_SIZE;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline ind_t
RSB_sa_size(ind_t const n)
{
  return !RSB_is_split(n) ? 0 : (n / RSB_SIZE) * (n / RSB_SIZE) - 1;
}

/*----------------------------------------------------------------------------*/
/*! http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2 */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline ind_t
RSB_next_pow2(ind_t v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
#ifdef EFIKA_WITH_LONG
  v |= v >> 32;
#endif
  return v + 1;
}

/*----------------------------------------------------------------------------*/
/*! http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2 */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline ind_t
RSB_size(ind_t nr, ind_t nc)
{
  nr = RSB_next_pow2(nr);
  nc = RSB_next_pow2(nc);
  return nr > nc ? nr : nc;
}

/*----------------------------------------------------------------------------*/
/*! Compute the smallest index of an element that is greater than or equal to a
 *  given value. */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline ind_t
RSB_bsearch(
  int   const row,
  ind_t const k,
  ind_t const * const restrict za,
  ind_t const n
)
{
  ind_t l = 0, u = n;

  while (l < u) {
    ind_t const m = l + (u - l) / 2;
    ind_t const r = RSB_row(za[m]);
    ind_t const c = RSB_col(za[m]);
    if ((row ? r : c) < k)
      l = m + 1;
    else
      u = m;
  }

  return l;
}

/*----------------------------------------------------------------------------*/
/*! Convert a matrix stored in z-major order to one in row-major order */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline void
RSB_rsbcsr(
  ind_t const n,
  ind_t const nnz,
  ind_t const * const restrict za,
  val_t const * const restrict arsb,
  ind_t       * const restrict ia,
  ind_t       * const restrict ja,
  val_t       * const restrict acsr
)
{
  memset(ia, 0, (n + 1) * sizeof(*ia));

  for (ind_t i = 0; i < nnz; i++)
    ia[RSB_row(za[i]) % n]++;

  for (ind_t i = 0, p = 0; i <= n; i++) {
    ind_t const t = ia[i];
    ia[i] = p;
    p += t;
  }

  for (ind_t i = 0; i < nnz; i++) {
    ind_t const r = RSB_row(za[i]) % n;
    ja[ia[r]]     = za[i];
    acsr[ia[r]++] = arsb[i];
  }
}

/*----------------------------------------------------------------------------*/
/*! Convert a matrix stored in z-major order to one in column-major order */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline void
RSB_rsbcsc(
  ind_t const n,
  ind_t const nnz,
  ind_t const * const restrict za,
  val_t const * const restrict arsb,
  ind_t       * const restrict ia,
  ind_t       * const restrict ja,
  val_t       * const restrict acsc
)
{
  memset(ia, 0, (n + 1) * sizeof(*ia));

  for (ind_t i = 0; i < nnz; i++)
    ia[RSB_col(za[i]) % n]++;

  for (ind_t i = 0, p = 0; i <= n; i++) {
    ind_t const t = ia[i];
    ia[i] = p;
    p += t;
  }

  for (ind_t i = 0; i < nnz; i++) {
    ind_t const c = RSB_col(za[i]) % n;
    ja[ia[c]]     = za[i];
    acsc[ia[c]++] = arsb[i];
  }
}

#ifdef __cplusplus
# ifdef undef_restrict
#   undef restrict
# endif
#endif

/*----------------------------------------------------------------------------*/
/*! Private API. */
/*----------------------------------------------------------------------------*/
#ifdef __cplusplus
extern "C" {
#endif

ind_t RSB_spgemm_cache(ind_t,
                       ind_t, ind_t const *, val_t const *,
                       ind_t, ind_t const *, val_t const *,
                       ind_t *, val_t *,
                       ind_t *, val_t *);

ind_t RSB_spgemm_merge(ind_t,
                       ind_t, ind_t const *, val_t const *,
                       ind_t, ind_t const *, val_t const *,
                       ind_t *, val_t *, val_t *);

ind_t
RSB_spgemm(int, ind_t,
           ind_t, ind_t, ind_t, ind_t const *, ind_t const *, val_t const *,
           ind_t, ind_t, ind_t, ind_t const *, ind_t const *, val_t const *,
           ind_t *, ind_t *, val_t *,
           ind_t *, val_t *);

#ifdef __cplusplus
}
#endif

#endif /* EFIKA_CORE_RSB_H */
