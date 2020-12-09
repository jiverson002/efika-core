/* SPDX-License-Identifier: MIT */
#include <stdlib.h>
#include <string.h>

#include "efika/core.h"

#include "efika/core/gc.h"
#include "efika/core/pp.h"
#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/*! Function to re-order rows of a matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_EXPORT int
Matrix_copy(Matrix const * const M, Matrix * const C)
{
  /* ...garbage collected function... */
  GC_func_init();

  if (!pp_all(M, C))
    return -1;

  ind_t const nr  = M->nr;
  ind_t const nnz = M->nnz;
  ind_t const * const m_ia   = M->ia;
  ind_t const * const m_ja   = M->ja;
  val_t const * const m_a    = M->a;
  ind_t const * const m_vsiz = M->vsiz;
  val_t const * const m_vwgt = M->vwgt;

  if (!pp_all(m_ia, m_ja))
    return -1;

  ind_t * const c_ia = GC_malloc((nr + 1) * sizeof(*c_ia));
  ind_t * const c_ja = GC_malloc(nnz * sizeof(*c_ja));
  val_t * c_a    = NULL;
  val_t * c_vwgt = NULL;
  ind_t * c_vsiz = NULL;

  memcpy(c_ia, m_ia, (nr + 1) * sizeof(*c_ia));
  memcpy(c_ja, m_ja, nnz * sizeof(*c_ja));

  if (m_a) {
    c_a = GC_malloc(nnz * sizeof(*c_a));
    memcpy(c_a, m_a, nnz * sizeof(*c_a));
  }
  if (m_vwgt) {
    c_vwgt = GC_malloc(nr * sizeof(*c_vwgt));
    memcpy(c_vwgt, m_vwgt, nr * sizeof(*c_vwgt));
  }
  if (m_vsiz) {
    c_vsiz = GC_malloc(nr * sizeof(*c_vsiz));
    memcpy(c_vsiz, m_vsiz, nnz * sizeof(*c_vsiz));
  }

  C->fmt  = M->fmt;
  C->sort = M->sort;
  C->symm = M->symm;
  C->nr   = nr;
  C->nc   = M->nc;
  C->nnz  = nnz;
  C->ia   = c_ia;
  C->ja   = c_ja;
  C->a    = c_a;
  C->vwgt = c_vwgt;
  C->vsiz = c_vsiz;

  return 0;
}
