/* SPDX-License-Identifier: MIT */
#include "efika/core/rsb.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
typedef uint8_t  i8;
typedef uint16_t i16;
typedef uint32_t i32;
typedef uint64_t i64;

/*----------------------------------------------------------------------------*/
/*! Convert a matrix stored in z-major order to one in row-major order */
/*----------------------------------------------------------------------------*/
static inline void
RSB_rsbcsr_8x8(
  ind_t const n,
  ind_t const nnz,
  ind_t const * const restrict za,
  val_t const * const restrict arsb,
  i8          * const restrict ia,
  i8          * const restrict ja,
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
    i8 const r = RSB_row(za[i]) % n;
    i8 const c = RSB_col(za[i]) % n;
    ja[ia[r]]     = c;
    acsr[ia[r]++] = arsb[i];
  }

  for (ind_t i = n; i > 0; i--)
    ia[i] = ia[i - 1];
  ia[0] = 0;
}

/*----------------------------------------------------------------------------*/
/*! Convert a matrix stored in z-major order to one in column-major order */
/*----------------------------------------------------------------------------*/
static inline void
RSB_rsbcsc_8x8(
  ind_t const n,
  ind_t const nnz,
  ind_t const * const restrict za,
  val_t const * const restrict arsb,
  i8          * const restrict ia,
  i8          * const restrict ja,
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
    ind_t const r = RSB_row(za[i]) % n;
    ind_t const c = RSB_col(za[i]) % n;
    ja[ia[c]]     = r;
    acsc[ia[c]++] = arsb[i];
  }

  for (ind_t i = n; i > 0; i--)
    ia[i] = ia[i - 1];
  ia[0] = 0;
}

/*----------------------------------------------------------------------------*/
/*! Convert a matrix stored in z-major order to one in row-major order */
/*----------------------------------------------------------------------------*/
static inline void
RSB_rsbcsr_16x16(
  ind_t const n,
  ind_t const nnz,
  ind_t const * const restrict za,
  val_t const * const restrict arsb,
  i16         * const restrict ia,
  i16         * const restrict ja,
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
    i16 const r = RSB_row(za[i]) % n;
    i16 const c = RSB_col(za[i]) % n;
    ja[ia[r]]     = c;
    acsr[ia[r]++] = arsb[i];
  }

  for (ind_t i = n; i > 0; i--)
    ia[i] = ia[i - 1];
  ia[0] = 0;
}

/*----------------------------------------------------------------------------*/
/*! Convert a matrix stored in z-major order to one in column-major order */
/*----------------------------------------------------------------------------*/
static inline void
RSB_rsbcsc_16x16(
  ind_t const n,
  ind_t const nnz,
  ind_t const * const restrict za,
  val_t const * const restrict arsb,
  i16         * const restrict ia,
  i16         * const restrict ja,
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
    ind_t const r = RSB_row(za[i]) % n;
    ind_t const c = RSB_col(za[i]) % n;
    ja[ia[c]]     = r;
    acsc[ia[c]++] = arsb[i];
  }

  for (ind_t i = n; i > 0; i--)
    ia[i] = ia[i - 1];
  ia[0] = 0;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline ind_t
RSB_spgemm_csr_csr_8x8(
  ind_t const n,
  i8    const * const restrict ia,
  i8    const * const restrict ja,
  val_t const * const restrict a,
  i8    const * const restrict ib,
  i8    const * const restrict jb,
  val_t const * const restrict b,
  i8          * const restrict ic,
  i8          * const restrict jc,
  val_t       * const restrict c,
  val_t       * const restrict spa
)
{
  ind_t nnz = 0;

  ic[0] = 0;
  for (i8 i = 0; i < n; i++) {
    for (i8 j = ia[i]; j < ia[i + 1]; j++) {
      i8    const x = ja[j];
      val_t const v = a[j];

      for (i8 k = ib[x]; k < ib[x + 1]; k++) {
        i8    const y = jb[k];
        val_t const w = b[k];

        if (0.0 == spa[y])
          jc[nnz++] = y;

        spa[y] += v * w;

        /* correct for floating-point underflow, when v * w == 0.0 */
        nnz -= (0.0 == spa[y]);
      }
    }

    ic[i + 1] = nnz;

    /* BLAS_vgthrz(ic[i + 1] - ic[i], spa, c + ic[i], jc + ic[i]); */
    for (i8 j = ic[i]; j < ic[i + 1]; j++) {
      c[j] = spa[jc[j]];
      spa[jc[j]] = 0.0;
    }
  }

  return nnz;
}

static inline ind_t
RSB_spgemm_csr_csr_16x16(
  i16   const n,
  i16   const * const restrict ia,
  i16   const * const restrict ja,
  val_t const * const restrict a,
  i16   const * const restrict ib,
  i16   const * const restrict jb,
  val_t const * const restrict b,
  i16         * const restrict ic,
  i16         * const restrict jc,
  val_t       * const restrict c,
  val_t       * const restrict spa
)
{
  ind_t nnz = 0;

  ic[0] = 0;
  for (i16 i = 0; i < n; i++) {
    for (i16 j = ia[i]; j < ia[i + 1]; j++) {
      i16   const x = ja[j];
      val_t const v = a[j];

      for (i16 k = ib[x]; k < ib[x + 1]; k++) {
        i16   const y = jb[k];
        val_t const w = b[k];

        if (0.0 == spa[y])
          jc[nnz++] = y;

        spa[y] += v * w;

        /* correct for floating-point underflow, when v * w == 0.0 */
        nnz -= (0.0 == spa[y]);
      }
    }

    ic[i + 1] = nnz;

    /* BLAS_vgthrz(ic[i + 1] - ic[i], spa, c + ic[i], jc + ic[i]); */
    for (i16 j = ic[i]; j < ic[i + 1]; j++) {
      c[j] = spa[jc[j]];
      spa[jc[j]] = 0.0;
    }
  }

  return nnz;
}

/*----------------------------------------------------------------------------*/
/*! Multiply matrix A with matrix B entirely in cache, storing the result in
 *  matrix C (not necessarily stored in cache). The results will be stored in
 *  row-major order in compressed-index format in C. This function takes as
 *  input the current number of non-zeros in C and returns the new number of
 *  non-zeros in C. */
/*----------------------------------------------------------------------------*/
__attribute((unused)) static inline ind_t
RSB_spgemm_cache_v1(
  ind_t const n,
  ind_t const annz,
  ind_t const * const restrict za,
  val_t const * const restrict a,
  ind_t const bnnz,
  ind_t const * const restrict zb,
  val_t const * const restrict b,
  ind_t       * const restrict zc,
  val_t       * const restrict c,
  val_t       * const restrict spa
)
{
  /* XXX: At this point, we know the following:
   *      - A and B are square sub-matrices with the same dimension.
   *      - A and B are both stored in compressed-index format.
   *      - Together, A and B's non-zero values will fit into cache.
   *      - It is likely that A or B has many more non-zeros than the other.
   */

  // FIXME: hard-code
  static i16   icache[8000];
  static val_t vcache[4000] = { 0.0 };

  ind_t cnnz;

  i16   * const ia_cache = icache;
  i16   * const ja_cache = ia_cache + n + 1;
  val_t * const a_cache  = vcache;

  i16   * const ib_cache = ja_cache;
  i16   * const jb_cache = ib_cache + n + 1;
  val_t * const b_cache  = a_cache + annz;

  i16   * const ic_cache = jb_cache;
  i16   * const jc_cache = ic_cache + n + 1;
  val_t * const c_cache  = b_cache + bnnz;

  if (n <= 256) { /* [  0,   256] */
    RSB_rsbcsr_8x8(n, annz, za, a, (void*)ia_cache, (void*)ja_cache, a_cache);
    RSB_rsbcsc_8x8(n, bnnz, zb, b, (void*)ib_cache, (void*)jb_cache, b_cache);

    cnnz = RSB_spgemm_csr_csr_8x8(n,
                                  (void*)ia_cache, (void*)ja_cache, a_cache,
                                  (void*)ib_cache, (void*)jb_cache, b_cache,
                                  (void*)ic_cache, (void*)jc_cache, c_cache,
                                  spa);

    // TODO: merge c cache into c
  } else {        /* (256, 65536] */
    RSB_rsbcsr_16x16(n, annz, za, a, ia_cache, ja_cache, a_cache);
    RSB_rsbcsc_16x16(n, bnnz, zb, b, ib_cache, jb_cache, b_cache);

    cnnz = RSB_spgemm_csr_csr_16x16(n,
                                    ia_cache, ja_cache, a_cache,
                                    ib_cache, jb_cache, b_cache,
                                    ic_cache, jc_cache, c_cache,
                                    spa);

    // TODO: merge c cache into c
  }

  return cnnz;

  (void)zc;
  (void)c;
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline ind_t
RSB_spgemm_csr_csr_v2(
  ind_t const n,
  ind_t const annz,
  ind_t const * const restrict za,
  val_t const * const restrict a,
  ind_t const * const restrict zb,
  val_t const * const restrict b,
  ind_t       * const restrict zc,
  val_t       * const restrict c,
  ind_t       * const restrict map,
  val_t       * const restrict spa
)
{
  ind_t cnnz = 0;

  for (ind_t i = 0; i < annz;) {
    ind_t const row = RSB_row(za[i]);
    ind_t nnz = 0;

    for (ind_t j = 0; RSB_row(za[i]) == row; i++) {
      ind_t const col = RSB_col(za[i]);
      val_t const v = a[i];

      /* fast-forward the columns */
      for (; RSB_row(zb[j]) < col; j++);

      for (; RSB_row(zb[j]) == col; j++) {
        ind_t const y = RSB_col(zb[j]) % n;
        val_t const w = b[j];

        if (0.0 == spa[y])
          map[nnz++] = zb[j];

        spa[y] += v * w;

        /* correct for floating-point underflow, when v * w == 0.0 */
        nnz -= (0.0 == spa[y]);
      }
    }

    /* XXX: Entries of /C/ are accumulated in temporary memory. The thinking
     *      here is that these memories will be stored *mostly* in cache. If
     *      that is the case, then *most* accumulation is done in cache. Then,
     *      updating the entries in /C/ are just writes, which can potentially
     *      be optimized with non-temporal memory stores, so as not to pollute
     *      the cache with entries that will only be read during a single row.
     *
     *      I do not think that the compiler will generate non-temporal stores
     *      for this, so it may need to be hard-coded using intrinsics. This
     *      should be explored after proper benchmarking is implemented.
     */
    for (ind_t j = 0; j < nnz; j++, cnnz++) {
      ind_t const z = map[j];
      ind_t const col = RSB_col(z);
      ind_t const y = col % n;
      zc[cnnz] = RSB_idx(row, col);
      c[cnnz] = spa[y];
      spa[y] = 0.0;
    }
  }

  return cnnz;
}

/*----------------------------------------------------------------------------*/
/*! Multiply matrix A with matrix B entirely in cache, storing the result in
 *  matrix C (not necessarily stored in cache). The results will be stored in
 *  row-major order in compressed-index format in C. This function takes as
 *  input the current number of non-zeros in C and returns the new number of
 *  non-zeros in C. */
/*----------------------------------------------------------------------------*/
ind_t
RSB_spgemm_cache_v2(
  ind_t const n,
  ind_t const annz,
  ind_t const * const restrict za,
  val_t const * const restrict a,
  ind_t const bnnz,
  ind_t const * const restrict zb,
  val_t const * const restrict b,
  ind_t       * const restrict zc,
  val_t       * const restrict c,
  ind_t       * const restrict itmp,
  val_t       * const restrict vtmp
)
{
  /* XXX: At this point, we know the following:
   *      - A and B are square sub-matrices with the same dimension.
   *      - A and B are both stored in compressed-index format.
   *      - Together, A and B's non-zero values will fit into cache.
   *      - It is likely that A or B has many more non-zeros than the other.
   *      - The number of entries of C that result from multiplying A with B is
   *        bounded above by n^2.
   */

  // FIXME: hard-code
  static ind_t icache[3002];
  static val_t vcache[3000];

  /* XXX: Allocation of cache
   *  +-----------+-----------+-----------+-----------+-----------
   *  | za (annz) | zb (bnnz) |  a (annz) |  b (bnnz) | ...
   *  +-----------+-----------+-----------+-----------+-----------
   *  |                icache |                vcache | ...
   *  +-----------+-----------+-----------+-----------+-----------
   *
   *  No part of /C/ is assumed to be allocated in cache. That said, the
   *  *icache* and *vcache* do not fill the entire cache. There is space is
   *  reserved for the accumulation of non-zero elements of /C/. This will need
   *  to be at most 2 * n entries, as rows of /A/ are processed row by row.
   *  Thus, any given row can produce at most n entries in /C/. The amount of
   *  space reserved in the cache is controlled by the parameter *alpha*. This
   *  parameter represents the fraction of n that will have reserved entries in
   *  the cache. The equation, cache size - alpha * 2 * n, will determine the
   *  amount of cache reserved for /A/ and /B/.
   */
  ind_t * const za_cache  = icache;
  val_t * const a_cache   = vcache;
  ind_t * const zb_cache  = za_cache + annz;
  val_t * const b_cache   = a_cache + annz;

  /* */
  RSB_rsbcsr(n, annz, za, a, itmp, za_cache, a_cache);
  za_cache[annz] = ~za_cache[annz - 1]; // sentinel value

  RSB_rsbcsr(n, bnnz, zb, b, itmp, zb_cache, b_cache);
  zb_cache[bnnz] = ~zb_cache[bnnz - 1]; // sentinel value

  /* */
  return RSB_spgemm_csr_csr_v2(n,
                               annz, za_cache, a_cache,
                                     zb_cache, b_cache,
                                     zc,       c,
                               itmp, vtmp);
}
