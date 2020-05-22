/* SPDX-License-Identifier: MIT */
#include <stdbool.h>
#include <stdlib.h>

#include "efika/core/blas.h"
#include "efika/core/rsb.h"

#define LEVEL      100
#define SPLIT_SIZE 50000

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

        /* correct for floating-point underflow, when v * w == 0.0 */
        nnz -= (0.0 == spa[y]);
      }
    }

    ic[i + 1] = nnz;

    BLAS_vgthrz(ic[i + 1] - ic[i], spa, c + ic[i], jc + ic[i]);
  }
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline bool
RSB_in_cache(ind_t const a_nnz, ind_t const b_nnz)
{
  // FIXME: hard-code
  //return (a_nnz + b_nnz) <= 4000;
  return (a_nnz + b_nnz) <= SPLIT_SIZE;
}

/*----------------------------------------------------------------------------*/
/*! Multiply matrix A with matrix B entirely in cache, storing the result in
 *  matrix C (not necessarily stored in cache).
 *
 *  The results will be stored in row-major order in compressed-index format in
 *  C.
 *
 *  @param c_nnz The number of non-zeros currently in C.
 *
 *  @return The new number of non-zeros in C.
 */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline ind_t
RSB_spgemm_csr_csc_v0( /* merge-like */
  ind_t const a_nnz,
  ind_t const * const restrict a_za,
  val_t const * const restrict a_a,
  ind_t const b_nnz,
  ind_t const * const restrict b_za,
  val_t const * const restrict b_a,
  ind_t       c_nnz,
  ind_t       * const restrict c_za,
  val_t       * const restrict c_a
)
{
  for (ind_t ii = 0, i = 0, k = 0; ii < a_nnz;) {
    /* get current row */
    ind_t const r = RSB_row(a_za[i]);

    for (ind_t j = 0; j < b_nnz;) {
      /* get current column */
      ind_t const c = RSB_col(b_za[j]);

      /* initialize result */
      val_t res = 0.0;

      /* merge-like dot product C_rc = <A_r*, B_*c> */
      for (i = ii; RSB_row(a_za[i]) == r && RSB_col(b_za[j]) == c;) {
        if (RSB_col(a_za[i]) < RSB_row(b_za[j]))
          i++;
        else if (RSB_col(a_za[i]) > RSB_row(b_za[j]))
          j++;
        else
          res += a_a[i++] * b_a[j++];
      }

      if (res > 0.0) {
        ind_t const x = RSB_idx(r, c);

        /* fast-forward k to next output element */
        for (; c_za[k] < x; k++);

        /* get output index */
        ind_t const o = c_za[k] == x ? k : (c_za[c_nnz] = x, c_nnz++);

        /* record result */
        c_a[o] += res;
      }

      /* fast-forward j to next column */
      for (; RSB_col(b_za[j]) == c; j++);
    }

    /* fast-forward ii to next row */
    for (ii = i; RSB_row(a_za[ii]) == r; ii++);
  }

  return c_nnz;
}

/*----------------------------------------------------------------------------*/
/*! Multiply matrix A with matrix B entirely in cache, storing the result in
 *  matrix C (not necessarily stored in cache).
 *
 *  The results will be stored in row-major order in compressed-index format in
 *  C.
 *
 *  @param c_nnz The number of non-zeros currently in C.
 *
 *  @return The new number of non-zeros in C.
 */
/*----------------------------------------------------------------------------*/
__attribute__((unused)) static inline ind_t
RSB_spgemm_csr_csc_v1( /* sparse-accumulator */
  ind_t const n,
  ind_t const a_nnz,
  ind_t const * const restrict a_za,
  val_t const * const restrict a_a,
  ind_t const b_nnz,
  ind_t const * const restrict b_za,
  val_t const * const restrict b_a,
  ind_t       c_nnz,
  ind_t       * const restrict c_za,
  val_t       * const restrict c_a,
  val_t       * const restrict spa
)
{
  for (ind_t i = 1; i < a_nnz; i++) {
    if (RSB_row(a_za[i]) == RSB_row(a_za[i - 1])) {
      if (RSB_col(a_za[i]) == RSB_col(a_za[i - 1])) {
        fprintf(stderr, "a.0\n"); abort();
      } else if (RSB_col(a_za[i]) < RSB_col(a_za[i - 1])) {
        fprintf(stderr, "a.1\n"); abort();
      }
    } else if (RSB_row(a_za[i]) < RSB_row(a_za[i - 1])) {
      fprintf(stderr, "a.2\n"); abort();
    }
  }
  for (ind_t i = 1; i < b_nnz; i++) {
    if (RSB_col(b_za[i]) == RSB_col(b_za[i - 1])) {
      if (RSB_row(b_za[i]) == RSB_row(b_za[i - 1])) {
        fprintf(stderr, "b.0\n"); abort();
      } else if (RSB_row(b_za[i]) < RSB_row(b_za[i - 1])) {
        fprintf(stderr, "b.1\n"); abort();
      }
    } else if (RSB_col(b_za[i]) < RSB_col(b_za[i - 1])) {
      fprintf(stderr, "b.2\n"); abort();
    }
  }
  for (ind_t i = 1; i < c_nnz; i++) {
    if (RSB_row(c_za[i]) == RSB_row(c_za[i - 1])) {
      if (RSB_col(c_za[i]) == RSB_col(c_za[i - 1])) {
        fprintf(stderr, "c.0\n"); abort();
      } else if (RSB_col(c_za[i]) < RSB_col(c_za[i - 1])) {
        fprintf(stderr, "c.1\n"); abort();
      }
    } else if (RSB_row(c_za[i]) < RSB_row(c_za[i - 1])) {
      fprintf(stderr, "c.2\n"); abort();
    }
  }

  for (ind_t i = 0, k = 0; i < a_nnz;) {
    ind_t const r = RSB_row(a_za[i]);

    for (ind_t j = i; RSB_row(a_za[j]) == r; j++)
      spa[RSB_col(a_za[j]) % n] = a_a[j];

    for (ind_t j = 0; j < b_nnz;) {
      /* get current column */
      ind_t const c = RSB_col(b_za[j]);

      /* initialize result */
      val_t res = 0.0;

      /* spa dot product C_rc = <A_r*, B_*c> */
      for (; RSB_col(b_za[j]) == c; j++)
        res += spa[RSB_row(b_za[j]) % n] * b_a[j];

      if (res > 0.0) {
        ind_t const x = RSB_idx(r, c);

        /* fast-forward k to next output element */
        for (; k < c_nnz && c_za[k] < x; k++);

        /* get output index */
        ind_t const o = k < c_nnz && c_za[k] == x
                      ? k
                      : (c_za[c_nnz] = x, c_nnz++);

        /* record result */
        c_a[o] += res;
      }
    }

    for (; RSB_row(a_za[i]) == r; i++)
      spa[RSB_col(a_za[i]) % n] = 0.0;
  }

  return c_nnz;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline ind_t
RSB_spgemm_csr_csc(
  ind_t const n,
  ind_t const a_nnz,
  ind_t const * const restrict a_za,
  val_t const * const restrict a_a,
  ind_t const b_nnz,
  ind_t const * const restrict b_za,
  val_t const * const restrict b_a,
  ind_t       c_nnz,
  ind_t       * const restrict c_za,
  val_t       * const restrict c_a,
  val_t       * const restrict spa
)
{
#if 0
  return RSB_spgemm_csr_csc_v0(a_nnz, a_za, a_a,
                               b_nnz, b_za, b_a,
                               c_nnz, c_za, c_a);
#else
  return RSB_spgemm_csr_csc_v1(n,
                               a_nnz, a_za, a_a,
                               b_nnz, b_za, b_a,
                               c_nnz, c_za, c_a,
                               spa);
#endif
}

/*----------------------------------------------------------------------------*/
/*! Multiply matrix A with matrix B entirely in cache, storing the result in
 *  matrix C (not necessarily stored in cache). The results will be stored in
 *  row-major order in compressed-index format in C. This function takes as
 *  input the current number of non-zeros in C and returns the new number of
 *  non-zeros in C. */
/*----------------------------------------------------------------------------*/
static inline ind_t
RSB_spgemm_cache(
  ind_t const n,
  ind_t const a_nnz,
  ind_t const * const restrict a_za,
  val_t const * const restrict a_a,
  ind_t const b_nnz,
  ind_t const * const restrict b_za,
  val_t const * const restrict b_a,
  ind_t       c_nnz,
  ind_t       * const restrict c_za,
  val_t       * const restrict c_a,
  ind_t       * const restrict tmp
)
{
  /* XXX: At this point, we know the following:
   *      - A and B are square sub-matrices with the same dimension.
   *      - A and B are both stored in compressed-index format.
   *      - Together, A and B's non-zero values will fit into cache.
   *      - It is likely that A or B has many more non-zeros than the other.
   */

  // FIXME: hard-code
  //static ind_t icache[4002];
  //static val_t vcache[4002];
  static ind_t * icache = NULL;
  if (!icache) icache = malloc((SPLIT_SIZE + 2) * sizeof(ind_t));
  static val_t * vcache = NULL;
  if (!vcache) vcache = malloc((SPLIT_SIZE + 2) * sizeof(val_t));
  static val_t * spa = NULL;
  if (!spa) spa = calloc((SPLIT_SIZE + 2), sizeof(val_t));

  ind_t * const a_za_cache = icache;
  val_t * const a_a_cache  = vcache;
  ind_t * const b_za_cache = icache + a_nnz + 1;
  val_t * const b_a_cache  = vcache + a_nnz + 1;

  /* */
  RSB_rsbcsr(n, a_nnz, a_za, a_a, tmp, a_za_cache, a_a_cache);
  a_za_cache[a_nnz] = ~a_za_cache[a_nnz - 1]; // sentinel value

  /* */
  RSB_rsbcsc(n, b_nnz, b_za, b_a, tmp, b_za_cache, b_a_cache);
  b_za_cache[b_nnz] = ~b_za_cache[b_nnz - 1]; // sentinel value

  /* */
  return RSB_spgemm_csr_csc(n,
                            a_nnz, a_za_cache, a_a_cache,
                            b_nnz, b_za_cache, b_a_cache,
                            c_nnz, c_za, c_a, spa);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static ind_t
RSB_spgemm(
  int   const lvl,
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
  ind_t       c_nnz,
  ind_t       * const restrict c_sa,
  ind_t       * const restrict c_za,
  val_t       * const restrict c_a,
  ind_t       * const restrict tmp
)
{
  if (lvl <= LEVEL) {
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "%6u x %6u // (%u, %u) (%u, %u) // %u, %u, %u\n", n, n,
            a_ro, a_co, b_ro, b_co, a_nnz, b_nnz, c_nnz);
  }

  /* shortcut if either matrix is all zeros */
  if (0 == a_nnz || 0 == b_nnz) {
    if (lvl <= LEVEL) {
      for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
      fprintf(stderr, "                0:> 0\n");
    }
    return 0;
  }

  /* check if multiplication can be done entirely in cache */
  if (!RSB_is_split(n) && RSB_in_cache(a_nnz, b_nnz)) {
    ind_t const nnz = RSB_spgemm_cache(n,
                                       a_nnz, a_za, a_a,  /* C = A * B */
                                       b_nnz, b_za, b_a,
                                       c_nnz, c_za, c_a,
                                       tmp) - c_nnz;
    if (lvl <= LEVEL) {
      for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
      fprintf(stderr, "                1:> %u\n", nnz);
    }
    return nnz;
  }

  /* temporary split values */
  ind_t a_sp[3];
  ind_t b_sp[3];
  ind_t c_sp[3];

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

    b_sp[0] = b_sa[0]; b_sp[1] = b_sa[1]; b_sp[2] = b_sa[2];
  } else {
    a_sp[1] = RSB_bsearch(1, a_rsp, a_za, a_nnz);
    a_sp[0] = RSB_bsearch(0, a_csp, a_za, a_sp[1]);
    a_sp[2] = a_sp[1] + RSB_bsearch(0, a_csp, a_za + a_sp[1], a_nnz - a_sp[1]);

    b_sp[1] = RSB_bsearch(1, b_rsp, b_za, b_nnz);
    b_sp[0] = RSB_bsearch(0, b_csp, b_za, b_sp[1]);
    b_sp[2] = b_sp[1] + RSB_bsearch(0, b_csp, b_za + b_sp[1], b_nnz - b_sp[1]);
  }

  /* compute size of book-keeping data structure */
  ind_t const nsa = RSB_sa_size(nn);

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
  ind_t const * const a11_sa = a_sa + 3;
  ind_t const * const a12_sa = a11_sa + nsa;
  ind_t const * const a21_sa = a12_sa + nsa;
  ind_t const * const a22_sa = a21_sa + nsa;
  ind_t const * const a11_za = a_za;
  ind_t const * const a12_za = a_za + a_sp[0];
  ind_t const * const a21_za = a_za + a_sp[1];
  ind_t const * const a22_za = a_za + a_sp[2];
  val_t const * const a11_a  = a_a;
  val_t const * const a12_a  = a_a + a_sp[0];
  val_t const * const a21_a  = a_a + a_sp[1];
  val_t const * const a22_a  = a_a + a_sp[2];

  /* compute /B/ quadrant arrays */
  ind_t const * const b11_sa = b_sa + 3;
  ind_t const * const b12_sa = b11_sa + nsa;
  ind_t const * const b21_sa = b12_sa + nsa;
  ind_t const * const b22_sa = b21_sa + nsa;
  ind_t const * const b11_za = b_za;
  ind_t const * const b12_za = b_za + b_sp[0];
  ind_t const * const b21_za = b_za + b_sp[1];
  ind_t const * const b22_za = b_za + b_sp[2];
  val_t const * const b11_a  = b_a;
  val_t const * const b12_a  = b_a + b_sp[0];
  val_t const * const b21_a  = b_a + b_sp[1];
  val_t const * const b22_a  = b_a + b_sp[2];

  /* compute /C/ quadrant arrays */
  ind_t * const c11_sa = c_sa + 3;
  ind_t * const c12_sa = c11_sa + nsa;
  ind_t * const c21_sa = c12_sa + nsa;
  ind_t * const c22_sa = c21_sa + nsa;

  if (lvl <= LEVEL) {
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "C11\n");
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "===\n");
  }
  /* C11 := A11 * B11 */
  ind_t nnz = RSB_spgemm(lvl + 1, nn,
                         a_ro,  a_co,  a11_nnz, a11_sa, a11_za, a11_a,
                         b_ro,  b_co,  b11_nnz, b11_sa, b11_za, b11_a,
                         0, c11_sa, c_za, c_a,
                         tmp);
  /* C11 += A12 * B21 */
  nnz += RSB_spgemm(lvl + 1, nn,
                    a_ro,  a_csp, a12_nnz, a12_sa, a12_za, a12_a,
                    b_rsp, b_co,  b21_nnz, b21_sa, b21_za, b21_a,
                    nnz, c11_sa, c_za, c_a,
                    tmp);
  c_sp[0] = nnz;

  if (lvl <= LEVEL) {
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "C12\n");
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "===\n");
  }
  /* C12 := A11 * B12 */
  nnz  = RSB_spgemm(lvl + 1, nn,
                    a_ro,  a_co,  a11_nnz, a11_sa, a11_za, a11_a,
                    b_ro,  b_csp, b12_nnz, b12_sa, b12_za, b12_a,
                    0, c12_sa, c_za + c_sp[0], c_a + c_sp[0],
                    tmp);
  /* C12 += A12 * B22 */
  nnz += RSB_spgemm(lvl + 1, nn,
                    a_ro,  a_csp, a12_nnz, a12_sa, a12_za, a12_a,
                    b_rsp, b_csp, b22_nnz, b22_sa, b22_za, b22_a,
                    nnz, c12_sa, c_za + c_sp[0], c_a + c_sp[0],
                    tmp);
  c_sp[1] = c_sp[0] + nnz;

  if (lvl <= LEVEL) {
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "C21\n");
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "===\n");
  }
  /* C21 := A21 * B11 */
  nnz  = RSB_spgemm(lvl + 1, nn,
                    a_rsp, a_co,  a21_nnz, a21_sa, a21_za, a21_a,
                    b_ro,  b_co,  b11_nnz, b11_sa, b11_za, b11_a,
                    0, c21_sa, c_za + c_sp[1], c_a + c_sp[1],
                    tmp);
  /* C21 += A22 * B21 */
  nnz += RSB_spgemm(lvl + 1, nn,
                    a_rsp, a_csp, a22_nnz, a22_sa, a22_za, a22_a,
                    b_rsp, b_co,  b21_nnz, b21_sa, b21_za, b21_a,
                    nnz, c21_sa, c_za + c_sp[1], c_a + c_sp[1],
                    tmp);
  c_sp[2] = c_sp[1] + nnz;

  if (lvl <= LEVEL) {
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "C22\n");
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "===\n");
  }
  /* C22 := A21 * B12 */
  nnz  = RSB_spgemm(lvl + 1, nn,
                    a_rsp, a_co,  a21_nnz, a21_sa, a21_za, a21_a,
                    b_ro,  b_csp, b12_nnz, b12_sa, b12_za, b12_a,
                    0, c22_sa, c_za + c_sp[2], c_a + c_sp[2],
                    tmp);
  /* C22 += A22 * B22 */
  nnz += RSB_spgemm(lvl + 1, nn,
                    a_rsp, a_csp, a22_nnz, a22_sa, a22_za, a22_a,
                    b_rsp, b_csp, b22_nnz, b22_sa, b22_za, b22_a,
                    nnz, c22_sa, c_za + c_sp[2], c_a + c_sp[2],
                    tmp);

  if (RSB_is_split(n)) {
    c_sa[0] = c_sp[0]; c_sa[1] = c_sp[1]; c_sa[2] = c_sp[2];
  }

  if (lvl <= LEVEL) {
    for (int i = 0; i < lvl; i++) fprintf(stderr, "  ");
    fprintf(stderr, "                2:> %u\n", c_sp[2] + nnz);
  }

  return c_sp[2] + nnz;
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
  ind_t       * const restrict cnnz,
  ind_t       * const restrict sc,
  ind_t       * const restrict zc,
  val_t       * const restrict c,
  ind_t       * const restrict tmp
)
{
  *cnnz = RSB_spgemm(0, n, 0, 0, annz, sa, za, a, 0, 0, bnnz, sb, zb, b, 0, sc, zc,
                     c, tmp);
}
