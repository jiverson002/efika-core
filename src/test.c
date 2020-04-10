/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core.h"

#include "efika/core/export.h"
#include "efika/core/gc.h"
#include "efika/core/pp.h"
#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/* helper macros */
/*----------------------------------------------------------------------------*/
#define fail()            return -1
#define fail_if(cond)     do { if (cond) { fail(); } } while (0)
#define fail_unless(cond) fail_if(!(cond))

/*----------------------------------------------------------------------------*/
/* helper functions for bsearch */
/*----------------------------------------------------------------------------*/
static int
_asc(void const * const a, void const * const b)
{
  return (*(ind_t*)a < *(ind_t*)b) ? -1
       : (*(ind_t*)a > *(ind_t*)b) ? 1
       : 0;
}

static int
_dsc(void const * const a, void const * const b)
{
  return (*(ind_t*)a > *(ind_t*)b) ? -1
       : (*(ind_t*)a < *(ind_t*)b) ? 1
       : 0;
}

/*----------------------------------------------------------------------------*/
/*! Function to test a matrix data structure for correctness. */
/*----------------------------------------------------------------------------*/
EFIKA_CORE_EXPORT int
Matrix_test(Matrix const * const M)
{
  fail_if(!M);

  /* unpack /M/ */
  int const fmt  = M->fmt;
  int const symm = M->symm;
  int const sort = M->sort;
  ind_t const nr   = M->nr;
  ind_t const nc   = M->nc;
  ind_t const nnz  = M->nnz;
  ind_t const ncon = M->ncon;
  ind_t const * const ia = M->ia;
  ind_t const * const ja = M->ja;
  val_t const * const a  = M->a;
  val_t const * const vwgt = M->vwgt;
  ind_t const * const vsiz = M->vsiz;

  if (  0 != fmt &&   1 != fmt &&  10 != fmt &&  11 != fmt && 100 != fmt &&
      101 != fmt && 110 != fmt && 111 != fmt)
    fail();

  fail_if(!pp_all(nr, nc, nnz, ia, ja));
  fail_if(has_adjwgt(fmt) && !a);
  fail_if(has_vtxwgt(fmt) && !pp_all(ncon, vwgt));
  fail_if(has_vtxsiz(fmt) && !vsiz);
  fail_if(NONE != sort && ASC != sort && DSC != sort);
  fail_if(0 != symm && (1 != symm || 0 != nnz % 2));

  for (ind_t i = 0; i < nr; i++) {
    /* vertex size is guaranteed to be >= 0 since ind_t is unsigned */

    for (ind_t j = 0; j < ncon; j++)
      fail_if(vwgt[i*ncon+j] < 0.0);

    for (ind_t j = ia[i]; j < ia[i + 1]; j++) {
      fail_if(ja[j] >= nc);

      if (has_adjwgt(fmt))
        fail_if(a[j] <= 0.0);

      if (j > ia[i]) {
        /* edges may not be equal --- simple graphs only */
        fail_if(ASC == sort && ja[j] <= ja[j - 1]);
        fail_if(DSC == sort && ja[j] >= ja[j - 1]);
      }

      if (1 == symm) {
        ind_t const jj = ja[j];
        ind_t const *idx = NULL;

        /* FIXME this should only be tested if the graph is simple. */
        fail_if(i == jj);

        if (ASC == sort) {
          idx = bsearch(&i, ja + ia[jj], ia[jj + 1] - ia[jj], sizeof(*ja), _asc);
        } else if (DSC == sort) {
          idx = bsearch(&i, ja + ia[jj], ia[jj + 1] - ia[jj], sizeof(*ja), _dsc);
          fail_if(!idx);
        }
        else {
          for (ind_t k = ia[jj]; k < ia[jj + 1]; k++) {
            if (i == ja[k]) {
              idx = ja + k;
              break;
            }
          }
        }

        /* edge weights must match in symmetrical matrix */
        if (has_adjwgt(fmt))
          fail_if(a[j] != a[(size_t)(idx - ja)]);
      }
    }
  }

  return 0;
}
