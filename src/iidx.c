/* SPDX-License-Identifier: MIT */
#include "efika/core/blas.h"
#include "efika/core/export.h"
#include "efika/core/gc.h"
#include "efika/core/rename.h"
#include "efika/core.h"

/*----------------------------------------------------------------------------*/
/*! Create inverted index.
 *
 * \param M Matrix
 * \param I Inverted index
 */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT int
Matrix_iidx(Matrix const * const M, Matrix * const I)
{
  /*==========================================================================*/
  GC_func_init();
  /*==========================================================================*/

  /* unpack /M/ */
  ind_t const         m_nr  = M->nr;
  ind_t const         m_nc  = M->nc;
  ind_t const         m_nnz = M->nnz;
  ind_t const * const m_ia  = M->ia;
  ind_t const * const m_ja  = M->ja;
  val_t const * const m_a   = M->a;

  /* allocate memory for inverted index */
  ind_t * const i_ia = GC_malloc((m_nc + 1) * sizeof(*i_ia));
  ind_t * const i_ja = GC_malloc(m_nnz * sizeof(*i_ja));
  val_t * const i_a  = GC_malloc(m_nnz * sizeof(*i_a));

  BLAS_csrcsc(m_nr, m_nc, m_ia, m_ja, m_a, i_ia, i_ja, i_a);

  /* record relevant info in /I/ */
  I->nr  = m_nc;
  I->nc  = m_nr;
  I->nnz = m_nnz;
  I->ia  = i_ia;
  I->ja  = i_ja;
  I->a   = i_a;

  return 0;
}
