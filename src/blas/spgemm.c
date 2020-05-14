/* SPDX-License-Identifier: MIT */
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "efika/core/blas.h"

/*----------------------------------------------------------------------------*/
/*! Sparse-sparse matrix multiplication C = A * B, where A is stored in CSR
 * format and B is stored in CSC format.
 *
 *  Algorithm:
 *    for each row i of A:
 *      load row i's values into a hash table (dense array)
 *      for each column of B:
 *        compute sparse dot product with hash table
 *        record result in C
 *      clear hash table
 */
/*----------------------------------------------------------------------------*/
void
BLAS_spgemm_csr_csc(
  ind_t const m,
  ind_t const p,
  ind_t const * const restrict ia,
  ind_t const * const restrict ja,
  val_t const * const restrict a,
  ind_t const * const restrict ib,
  ind_t const * const restrict jb,
  val_t const * const restrict b,
  ind_t       * const restrict ic,
  ind_t       * const restrict jc,
  val_t       * const restrict c,
  val_t       * const restrict h
)
{
  ic[0] = 0;
  for (ind_t i = 0, nnz = 0; i < m; i++) {
    BLAS_vsctr(ia[i + 1] - ia[i], a + ia[i], ja + ia[i], h);

    for (ind_t j = 0; j < p; j++) {
      c[nnz] = BLAS_vdoti(ib[j + 1] - ib[j], b + ib[j], jb + ib[j], h);
      if (c[nnz] > 0.0)
        jc[nnz++] = j;
    }

    BLAS_vsctrz(ia[i + 1] - ia[i], ja + ia[i], h);

    ic[i + 1] = nnz;
  }
}

/*----------------------------------------------------------------------------*/
/*! Sparse-sparse matrix multiplication C = A * B, where A and B are stored in
 *  CSR format and B is treated as an inverted index.
 *
 *  Algorithm:
 *    for each row i in A:
 *      for each non-zero column x with value v:
 *        for each row y in B that has a non-zero in column x with value w:
 *          accumulate the multiplication v * w in hash table (dense array) for
 *          key y
 *      gather results from hash table into C
 *      clear hash table
 */
/*----------------------------------------------------------------------------*/
void
BLAS_spgemm_csr_csr(
  ind_t const m,
  ind_t const * const restrict ia,
  ind_t const * const restrict ja,
  val_t const * const restrict a,
  ind_t const * const restrict ib,
  ind_t const * const restrict jb,
  val_t const * const restrict b,
  ind_t       * const restrict ic,
  ind_t       * const restrict jc,
  val_t       * const restrict c,
  val_t       * const restrict spa
)
{
  ic[0] = 0;
  for (ind_t i = 0, nnz = 0; i < m; i++) {
    for (ind_t j = ia[i]; j < ia[i + 1]; j++) {
      ind_t const x = ja[j];
      val_t const v = a[j];

      for (ind_t k = ib[x]; k < ib[x + 1]; k++) {
        ind_t const y = jb[k];
        val_t const w = b[k];

        if (0.0 == spa[y])
          jc[nnz++] = y;

        spa[y] += v * w;

        /* correct floating-point underflow, when v * w == 0.0 */
        nnz -= (0.0 == spa[y]);
      }
    }

    ic[i + 1] = nnz;

    BLAS_vgthrz(ic[i + 1] - ic[i], spa, c + ic[i], jc + ic[i]);
  }
}

/*----------------------------------------------------------------------------*/
/*! Convert a matrix stored in z-major order to one in row-major order */
/*----------------------------------------------------------------------------*/
static inline void
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
  ind_t const half = sizeof(ind_t) * CHAR_BIT / 2;

  memset(ia, 0, (n + 1) * sizeof(*ia));

  for (ind_t i = 0; i < nnz; i++)
    ia[za[i] >> half]++;

  for (ind_t i = 0, p = 0; i <= n; i++) {
    ind_t const t = ia[i];
    ia[i] = p;
    p += t;
  }

  for (ind_t i = 0; i < nnz; i++) {
    ind_t const r = za[i] >> half;
    ja[ia[r]]     = za[i];
    acsr[ia[r]++] = arsb[i];
  }
}

/*----------------------------------------------------------------------------*/
/*! Convert a matrix stored in z-major order to one in column-major order */
/*----------------------------------------------------------------------------*/
static inline void
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
  ind_t const half = sizeof(ind_t) * CHAR_BIT / 2;
  ind_t const mask = ((ind_t)-1) >> half;

  memset(ia, 0, (n + 1) * sizeof(*ia));

  for (ind_t i = 0; i < nnz; i++)
    ia[za[i] & mask]++;

  for (ind_t i = 0, p = 0; i <= n; i++) {
    ind_t const t = ia[i];
    ia[i] = p;
    p += t;
  }

  for (ind_t i = 0; i < nnz; i++) {
    ind_t const c = za[i] & mask;
    ja[ia[c]]     = za[i];
    acsc[ia[c]++] = arsb[i];
  }
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline ind_t
h(ind_t const k, ind_t * const restrict hm)
{
  /* We will concentrate values in the first HM_SIZE slots. Once those have
   * filled, values will be placed in order starting at the end of the hash map.
   * */
  // FIXME: hard-code
  ind_t const HM_SIZE = 2048;
  ind_t const HM_UNKNOWN = (ind_t)-1;

  /* Compute the slot */
  ind_t x = k & (HM_SIZE - 1);

  /* If the slot is empty, populate it with the key */
  if (HM_UNKNOWN == hm[x])
    hm[x] = k;

  /* If we have the slot, return it. */
  if (k == hm[x])
    return x;

  /* If the slot was filled with another value, use linear probing to find the
   * the value or the next available slot. */
  for (ind_t i = 1; i < HM_SIZE; i++) {
    ind_t const y = (x + i) % HM_SIZE;
    if (HM_UNKNOWN == hm[y])
      hm[y] = k;
    if (k == hm[y])
      return y;
  }

  /* If table is completely full, just starting filling in values in order in
   * the locations after the hash table. */
  for (ind_t i = HM_SIZE; ; i++) {
    if (HM_UNKNOWN == hm[i])
      hm[i] = k;
    if (k == hm[i])
      return i;
  }

  /* error: should never reach here. */
  return (ind_t)-1;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline bool
RSB_is_split(ind_t const n)
{
  return n > ((ind_t)1 << (sizeof(ind_t) * CHAR_BIT / 2));
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline bool
RSB_in_cache(ind_t const a_nnz, ind_t const b_nnz)
{
  // FIXME: hard-code
  return (a_nnz + b_nnz) <= 4000;
}

/*----------------------------------------------------------------------------*/
/*! Compute the smallest index of an element that is greater than or equal to a
 *  given value. */
/*----------------------------------------------------------------------------*/
static inline ind_t
RSB_bsearch(
  int   const row,
  ind_t const k,
  ind_t const * const restrict za,
  ind_t const n
)
{
  ind_t const half = sizeof(ind_t) * CHAR_BIT / 2;
  ind_t const mask = ((ind_t)-1) >> half;
  ind_t l = 0, u = n;

  while (l < u) {
    ind_t const m = l + (u - l) / 2;
    ind_t const r = (za[m] >> half);
    ind_t const c = (za[m] &  mask);
    if ((row ? r : c) < k)
      l = m + 1;
    else
      u = m;
  }

  return l;
}

/*----------------------------------------------------------------------------*/
/*! Multiply matrix A (stored in compressed row-major order) with matrix B
 *  (stored in compressed column-major order), storing the result in C, which is
 *  a sparse accumulator (hash table). */
/*----------------------------------------------------------------------------*/
static inline void
RSB_spgemm_csr_csc(
  ind_t const a_nnz,
  ind_t const * const restrict a_ja,
  val_t const * const restrict a_a,
  ind_t const b_nnz,
  ind_t const * const restrict b_ja,
  val_t const * const restrict b_a,
  ind_t       * const restrict c_za,
  val_t       * const restrict c_a
)
{
#define ZSHIFT     (sizeof(ind_t) * CHAR_BIT / 2)
#define ZMASK      (((ind_t)-1) >> ZSHIFT)
#define zrow(z)    ((z) >> ZSHIFT)
#define zcol(z)    ((z) &  ZMASK)
#define zidx(r, c) ((r) << ZSHIFT | (c))

  for (ind_t ii = 0, i = 0, c_nnz = 0; ii < a_nnz;) {
    /* get current row */
    ind_t const r = zrow(a_ja[i]);

    for (ind_t j = 0; j < b_nnz;) {
      /* get current column */
      ind_t const c = zcol(b_ja[j]);

      /* merge-like dot product C_rc = <A_r*, B_*c> */
      val_t res = 0.0;
      for (i = ii; zrow(a_ja[i]) == r && zcol(b_ja[j]) == c;) {
        if (zcol(a_ja[i]) < zrow(b_ja[j]))
          i++;
        else if (zcol(a_ja[i]) > zrow(b_ja[j]))
          j++;
        else
          res += a_a[i++] * b_a[j++];
      }

      /* record result */
      if (res > 0.0) {
        c_za[c_nnz]  = zidx(r, c);
        c_a[c_nnz++] = res;
      }

      /* fast-forward j to next column */
      for (; zcol(b_ja[j]) == c; j++);
    }

    /* fast-forward ii to next row */
    for (ii = i; zrow(a_ja[ii]) == r; ii++);
  }

#undef ZSHIFT
#undef ZMASK
#undef zrow
#undef zcol
#undef zidx
}

/*----------------------------------------------------------------------------*/
/*! Multiply matrix A (stored in compressed column-major order) with matrix B
 *  (stored in compressed row-major order), storing the result in C, which is a
 *  sparse accumulator (hash table). */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline void
RSB_spgemm_csc_csr(
  ind_t const a_nnz,
  ind_t const * const restrict a_ja,
  val_t const * const restrict a_a,
  ind_t const b_nnz,
  ind_t const * const restrict b_ja,
  val_t const * const restrict b_a,
  ind_t       * const restrict c_za,
  val_t       * const restrict c_a
)
{
  ind_t const half = sizeof(ind_t) * CHAR_BIT / 2;
  ind_t const mask = ((ind_t)-1) >> half;

  for (ind_t i = 0, j = 0; i < a_nnz;) {
    /* fast-forward to the next column */
    ind_t const c1 = a_ja[i] & mask;

    /* fast-forward b to row c1 */
    for (; j < b_nnz && (b_ja[j] >> half) < c1; j++);

    /* for each row of A with non-zero in column c1 */
    for (; i < a_nnz && (a_ja[i] & mask) == c1; i++) {
      ind_t const r = a_ja[i] >> half;
      val_t const v = a_a[i];

      /* for each column of B with non-zero in row c1 */
      for (ind_t k = j; k < b_nnz && (b_ja[k] >> half) == c1; k++) {
        ind_t const z = (r << half) | (b_ja[k] & mask);
        ind_t const x = h(z, c_za);
        c_a[x] += v * b_a[k];
      }
    }
  }
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline void
RSB_spgemm_cache(
  ind_t const n,
  ind_t const a_ro,
  ind_t const a_co,
  ind_t const a_nnz,
  ind_t const * const restrict a_za,
  val_t const * const restrict a_a,
  ind_t const b_ro,
  ind_t const b_co,
  ind_t const b_nnz,
  ind_t const * const restrict b_za,
  val_t const * const restrict b_a,
  ind_t       * const restrict c_za,
  val_t       * const restrict c_a,
  ind_t       * const restrict ia
)
{
  /* XXX: At this point, we know the following:
   *      - A and B are square sub-matrices with the same dimension.
   *      - A and B are both stored in compressed-index format.
   *      - Together, A and B's non-zero values will fit into cache.
   *      - It is likely that A or B has many more non-zeros than the other.
   */

  // FIXME: hard-code
  static ind_t icache[4002];
  static val_t vcache[4002];

  ind_t * const a_ja_cache = icache;
  val_t * const a_a_cache  = vcache;
  ind_t * const b_ja_cache = icache + a_nnz + 1;
  val_t * const b_a_cache  = vcache + a_nnz + 1;

  /* */
  RSB_rsbcsr(n, a_nnz, a_za, a_a, ia, a_ja_cache, a_a_cache);
  a_ja_cache[a_nnz] = ~a_ja_cache[a_nnz - 1]; // sentinel value
  /* */
  RSB_rsbcsc(n, b_nnz, b_za, b_a, ia, b_ja_cache, b_a_cache);
  b_ja_cache[b_nnz] = ~b_ja_cache[b_nnz - 1]; // sentinel value
  /* */
  RSB_spgemm_csr_csc(a_nnz, a_ja_cache, a_a_cache,
                     b_nnz, b_ja_cache, b_a_cache,
                     c_za, c_a);

  (void)a_ro;
  (void)a_co;
  (void)b_ro;
  (void)b_co;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static void
RSB_spgemm(
  ind_t const n,
  ind_t const a_ro,
  ind_t const a_co,
  ind_t const a_nnz,
  ind_t const * const restrict a_sa,
  ind_t const * const restrict a_za,
  val_t const * const restrict a_a,
  ind_t const b_ro,
  ind_t const b_co,
  ind_t const b_nnz,
  ind_t const * const restrict b_sa,
  ind_t const * const restrict b_za,
  val_t const * const restrict b_a,
  ind_t       * const restrict c_sa,
  ind_t       * const restrict c_za,
  val_t       * const restrict c_a,
  ind_t       * const restrict tmp
)
{
  /* shortcut if either matrix is all zeros */
  if (0 == a_nnz || 0 == b_nnz)
    return;

  /* check if multiplication can be done entirely in cache */
  if (!RSB_is_split(n) && RSB_in_cache(a_nnz, b_nnz)) {
    RSB_spgemm_cache(n,
                     a_ro, a_co, a_nnz, a_za, a_a,  /* C = A * B */
                     b_ro, b_co, b_nnz, b_za, b_a,
                     c_za, c_a,
                     tmp);
    return;
  }

  abort();

  /* temporary split values */
  ind_t a_sp[6] = { 0 };
  ind_t b_sp[6] = { 0 };

  /* compute quadrant dimensions */
  ind_t const nn = n / 2;

  /* compute quadrant offsets */
  ind_t const a_rsp = a_ro + nn;
  ind_t const a_csp = a_co + nn;
  ind_t const b_rsp = b_ro + nn;
  ind_t const b_csp = b_co + nn;

  /* determine if splits are stored implicitly or explicitly */
  if (RSB_is_split(n)) {
    a_sp[0] = a_sa[0]; a_sp[1] = a_sa[1]; a_sp[2] = a_sa[2];
    a_sp[3] = a_sa[3]; a_sp[4] = a_sa[4]; a_sp[5] = a_sa[5];

    b_sp[0] = b_sa[0]; b_sp[1] = b_sa[1]; b_sp[2] = b_sa[2];
    b_sp[3] = b_sa[3]; b_sp[4] = b_sa[4]; b_sp[5] = b_sa[5];
  } else {
    a_sp[1] = RSB_bsearch(1, a_rsp, a_za, a_nnz);
    a_sp[0] = RSB_bsearch(0, a_csp, a_za, a_sp[1]);
    a_sp[2] = a_sp[1] + RSB_bsearch(0, a_csp, a_za + a_sp[1], a_nnz - a_sp[1]);

    b_sp[1] = RSB_bsearch(1, b_rsp, b_za, b_nnz);
    b_sp[0] = RSB_bsearch(0, b_csp, b_za, b_sp[1]);
    b_sp[2] = b_sp[1] + RSB_bsearch(0, b_csp, b_za + b_sp[1], b_nnz - b_sp[1]);
  }

  /* compute /A/ quadrant # non-zeros */
  ind_t const a11_nnz = a_sp[0];
  ind_t const a12_nnz = a_sp[1] - a_sp[0];
  ind_t const a21_nnz = a_sp[2] - a_sp[1];
  ind_t const a22_nnz = a_nnz   - a_sp[2];

  /* compute /B/ quadrant # non-zeros */
  ind_t const b11_nnz = b_sp[0];
  ind_t const b12_nnz = b_sp[1] - b_sp[0];
  ind_t const b21_nnz = b_sp[2] - b_sp[1];
  ind_t const b22_nnz = b_nnz   - b_sp[2];

  /* compute /A/ quadrant arrays */
  ind_t const * const a11_sa = a_sa + 6;
  ind_t const * const a12_sa = a_sa + a_sp[3];
  ind_t const * const a21_sa = a_sa + a_sp[4];
  ind_t const * const a22_sa = a_sa + a_sp[5];
  ind_t const * const a11_za = a_za;
  ind_t const * const a12_za = a_za + a_sp[0];
  ind_t const * const a21_za = a_za + a_sp[1];
  ind_t const * const a22_za = a_za + a_sp[2];
  val_t const * const a11_a  = a_a;
  val_t const * const a12_a  = a_a ? a_a + a_sp[0] : NULL;
  val_t const * const a21_a  = a_a ? a_a + a_sp[1] : NULL;
  val_t const * const a22_a  = a_a ? a_a + a_sp[2] : NULL;

  /* compute /B/ quadrant arrays */
  ind_t const * const b11_sa = b_sa + 6;
  ind_t const * const b12_sa = b_sa + b_sp[3];
  ind_t const * const b21_sa = b_sa + b_sp[4];
  ind_t const * const b22_sa = b_sa + b_sp[5];
  ind_t const * const b11_za = b_za;
  ind_t const * const b12_za = b_za + b_sp[0];
  ind_t const * const b21_za = b_za + b_sp[1];
  ind_t const * const b22_za = b_za + b_sp[2];
  val_t const * const b11_a  = b_a;
  val_t const * const b12_a  = b_a ? b_a + b_sp[0] : NULL;
  val_t const * const b21_a  = b_a ? b_a + b_sp[1] : NULL;
  val_t const * const b22_a  = b_a ? b_a + b_sp[2] : NULL;

  /* recursively multiply quadrants */
  RSB_spgemm(nn,                                           // C11 := A11 * B11
             a_ro,  a_co,  a11_nnz, a11_sa, a11_za, a11_a,
             b_ro,  b_co,  b11_nnz, b11_sa, b11_za, b11_a,
             c_sa, c_za, c_a,
             tmp);
  RSB_spgemm(nn,                                           // C11 += A12 * B21
             a_ro,  a_csp, a12_nnz, a12_sa, a12_za, a12_a,
             b_rsp, b_co,  b21_nnz, b21_sa, b21_za, b21_a,
             c_sa, c_za, c_a,
             tmp);
  RSB_spgemm(nn,                                           // C12 := A11 * B12
             a_ro,  a_co,  a11_nnz, a11_sa, a11_za, a11_a,
             b_ro,  b_csp, b12_nnz, b12_sa, b12_za, b12_a,
             c_sa, c_za, c_a,
             tmp);
  RSB_spgemm(nn,                                           // C12 += A12 * B22
             a_ro,  a_csp, a12_nnz, a12_sa, a12_za, a12_a,
             b_rsp, b_csp, b22_nnz, b22_sa, b22_za, b22_a,
             c_sa, c_za, c_a,
             tmp);
  RSB_spgemm(nn,                                           // C21 := A21 * B11
             a_rsp, a_co,  a21_nnz, a21_sa, a21_za, a21_a,
             b_ro,  b_co,  b11_nnz, b11_sa, b11_za, b11_a,
             c_sa, c_za, c_a,
             tmp);
  RSB_spgemm(nn,                                           // C21 += A22 * B21
             a_rsp, a_csp, a22_nnz, a22_sa, a22_za, a22_a,
             b_rsp, b_co,  b21_nnz, b21_sa, b21_za, b21_a,
             c_sa, c_za, c_a,
             tmp);
  RSB_spgemm(nn,                                           // C22 := A21 * B12
             a_rsp, a_co,  a21_nnz, a21_sa, a21_za, a21_a,
             b_ro,  b_csp, b12_nnz, b12_sa, b12_za, b12_a,
             c_sa, c_za, c_a,
             tmp);
  RSB_spgemm(nn,                                           // C22 += A22 * B22
             a_rsp, a_csp, a22_nnz, a22_sa, a22_za, a22_a,
             b_rsp, b_csp, b22_nnz, b22_sa, b22_za, b22_a,
             c_sa, c_za, c_a,
             tmp);
}

/*----------------------------------------------------------------------------*/
/*! Sparse-sparse matrix multiplication C = A * B, where A and B are stored in
 *  RSB format.
 */
/*----------------------------------------------------------------------------*/
void
BLAS_spgemm_rsb_rsb(
  ind_t const n,
  ind_t const annz,
  ind_t const * const restrict sa,
  ind_t const * const restrict za,
  val_t const * const restrict a,
  ind_t const bnnz,
  ind_t const * const restrict sb,
  ind_t const * const restrict zb,
  val_t const * const restrict b,
  ind_t       * const restrict sc,
  ind_t       * const restrict zc,
  val_t       * const restrict c,
  ind_t       * const restrict tmp
)
{
  RSB_spgemm(n, 0, 0, annz, sa, za, a, 0, 0, bnnz, sb, zb, b, sc, zc, c, tmp);
}
