/* SPDX-License-Identifier: MIT */
#include "efika/core/rsb.h"

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
static inline ind_t
RSB_spgemm_csr_csr_v2(
  ind_t const n,
  ind_t const annz,
  ind_t const * const restrict za,
  val_t const * const restrict a,
  ind_t const bnnz,
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

    for (ind_t j = 0; i < annz && RSB_row(za[i]) == row; i++) {
      ind_t const col = RSB_col(za[i]);
      val_t const v = a[i];

      /* fast-forward the columns */
      for (; j < bnnz && RSB_row(zb[j]) < col; j++);

      if (j == bnnz)
        break;

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

    /* make sure that /A/ is advanced to next row */
    for (; i < annz && RSB_row(za[i]) == row; i++);

    /* XXX: Entries of /C/ are accumulated in temporary memory. The thinking
     *      here is that these memories will be stored *mostly* in cache. If
     *      that is the case, then *most* accumulation is done in cache. Then,
     *      updating the entries in /C/ are just writes, which can potentially
     *      be optimized with non-temporal memory stores, so as not to pollute
     *      the cache with entries that will only be read during a single row.
     *
     * TODO: I do not think that the compiler will generate non-temporal stores
     *       for this, so it may need to be hard-coded using intrinsics. This
     *       should be explored after proper benchmarking is implemented.
     */
    for (ind_t j = 0; j < nnz; j++, cnnz++) {
      ind_t const col = RSB_col(map[j]);
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
 *  row-major order in compressed-index format in C. */
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
  static ind_t icache[4000];
  static val_t vcache[4000];

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
  //za_cache[annz] = ~za_cache[annz - 1]; // sentinel value

  /* */
  RSB_rsbcsr(n, bnnz, zb, b, itmp, zb_cache, b_cache);
  //zb_cache[bnnz] = ~zb_cache[bnnz - 1]; // sentinel value

  /* */
  return RSB_spgemm_csr_csr_v2(n,
                               annz, za_cache, a_cache,
                               bnnz, zb_cache, b_cache,
                                     zc,       c,
                               itmp, vtmp);
}

/*----------------------------------------------------------------------------*/
/*! */
/*----------------------------------------------------------------------------*/
ind_t
RSB_spgemm_merge_v2(
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
  ind_t cnnz = 0;

  for (ind_t i = 0, j = 0; i < annz || j < bnnz;) {
    if (j == bnnz || RSB_row(za[i]) < RSB_row(zb[j])) {
      zc[cnnz] = za[i];
      c[cnnz++] = a[i++];
    } else if (i == annz || RSB_row(za[i]) > RSB_row(zb[j])) {
      zc[cnnz] = zb[j];
      c[cnnz++] = b[j++];
    } else {
      /* TODO: Improve spatial locality of sparse-accumulator accesses --- most
       *       likely via a hash table. */

      ind_t const row = RSB_row(za[i]);

      /* populate sparse-accumulator will values from row of block A */
      for (ind_t k = i; k < annz && RSB_row(za[k]) == row; k++)
        spa[RSB_col(za[k]) % n] = a[k];

      /* merge values from row of block B with sparse-accumulator */
      for (; j < bnnz && RSB_row(zb[j]) == row; j++) {
        ind_t const y = RSB_col(zb[j]) % n;
        zc[cnnz] = zb[j];
        c[cnnz++] = b[j] + spa[y];
        spa[y] = 0.0;
      }

      /* advance to next row, clearing spa and populating results as we go */
      for (; i < annz && RSB_row(za[i]) == row; i++) {
        ind_t const y = RSB_col(za[i]) % n;
        if (spa[y] > 0.0) {
          zc[cnnz] = za[i];
          c[cnnz++] = a[i];
          spa[y] = 0.0;
        }
      }
    }
  }

  return cnnz;
}
