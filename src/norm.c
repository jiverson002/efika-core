/* SPDX-License-Identifier: MIT */
#include "efika/core.h"

#include "efika/core/blas.h"
#include "efika/core/pp.h"
#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/*! Normalizes the rows of the matrix to be unit length. */
/*----------------------------------------------------------------------------*/
EFIKA_EXPORT int
Matrix_norm(Matrix * const M)
{
  /* unpack /M/ */
  ind_t const         nr = M->nr;
  ind_t const * const ia = M->ia;
  val_t       * const a  = M->a;

  /* validate input */
  if (!pp_all(nr, ia, a))
    return -1;

  /* l2-normalize each row */
  for (ind_t i = 0; i < nr; i++) {
    val_t const norm = BLAS_vnrm2(ia[i + 1] - ia[i], a + ia[i]);
    if (norm > 0) {
      val_t const alpha = (val_t)(1.0 / norm);
      BLAS_vscal(ia[i + 1] - ia[i], alpha, a + ia[i]);
    }
  }

  return 0;
}
