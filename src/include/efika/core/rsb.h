/* SPDX-License-Identifier: MIT */
#ifndef EFIKA_CORE_RSB_H
#define EFIKA_CORE_RSB_H 1

#include <limits.h>
#include <stdbool.h>

#include "efika/core.h"

#include "efika/core/rename.h"

#ifdef __cplusplus
# ifndef restrict
#   define undef_restrict
#   define restrict
# endif
#endif

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

#ifdef __cplusplus
# ifdef undef_restrict
#   undef restrict
# endif
#endif

#endif /* EFIKA_CORE_RSB_H */
