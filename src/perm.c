/* SPDX-License-Identifier: MIT */
#include <stdbool.h>
#include <stdlib.h>

#include "efika/core.h"

#include "efika/core/export.h"
#include "efika/core/gc.h"
#include "efika/core/pp.h"
#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/*! Function to permute the rows of a matrix. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT int
Matrix_perm(Matrix * const M, ind_t * perm, ind_t * iperm)
{
  /* ...garbage collected function... */
  GC_func_init();

  bool freeperm = false, freeiperm = false;

  if (!pp_any(perm, iperm))
    return -1;

  int   const symm = M->symm;
  ind_t const nr   = M->nr;
  ind_t const nnz  = M->nnz;
  ind_t const * const ia   = M->ia;
  ind_t const * const ja   = M->ja;
  ind_t const * const vsiz = M->vsiz;
  val_t const * const a    = M->a;
  val_t const * const vwgt = M->vwgt;

  if (!pp_all(ia, ja))
    return -1;

  ind_t * const nia = GC_malloc((nr + 1) * sizeof(*nia));
  ind_t * const nja = GC_malloc(nnz * sizeof(*nja));
  val_t * na = NULL, * nvwgt = NULL;
  ind_t * nvsiz = NULL;
  if (a)
    na = GC_malloc(nnz * sizeof(*na));
  if (vwgt)
    nvwgt = GC_malloc(nr * sizeof(*nvwgt));
  if (vsiz)
    nvsiz = GC_malloc(nr * sizeof(*nvsiz));

  if (!perm) {
    freeperm = true;
    perm = GC_malloc(nr * sizeof(*perm));
    for (ind_t i = 0; i < nr; i++)
      perm[iperm[i]] = i;
  }
  if (!iperm) {
    freeiperm = true;
    iperm = GC_malloc(nr * sizeof(*iperm));
    for (ind_t i = 0; i < nr; i++)
      iperm[perm[i]] = i;
  }

  nia[0] = 0;
  for (ind_t i = 0, nnnz = 0; i < nr; i++) {
    ind_t const ii = iperm[i];

    for (ind_t j = ia[ii]; j < ia[ii + 1]; j++) {
      if (a)
        na[nnnz] = a[j];
      nja[nnnz++] = 1 == symm ? perm[ja[j]] : ja[j];
    }

    if (vwgt)
      nvwgt[i] = vwgt[ii];
    if (vsiz)
      nvsiz[i] = vsiz[ii];

    nia[i + 1] = nnnz;
  }
  GC_assert(nnz == nia[nr]);

  if (1 == symm)
    M->sort = NONE;
  M->ia   = nia;
  M->ja   = nja;
  M->a    = na;
  M->vwgt = nvwgt;
  M->vsiz = nvsiz;

  unsafe_free((void*)ia);
  unsafe_free((void*)ja);
  unsafe_free((void*)a);
  unsafe_free((void*)vwgt);
  unsafe_free((void*)vsiz);

  if (freeperm)
    GC_free(perm);
  if (freeiperm)
    GC_free(iperm);

  return 0;
}
