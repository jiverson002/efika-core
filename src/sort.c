/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core.h"

#include "efika/core/pp.h"
#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/* helper structs for sorting                                                 */
/*----------------------------------------------------------------------------*/
struct iv
{
  ind_t k;
  val_t v;
};

struct vi
{
  val_t k;
  ind_t v;
};

/*----------------------------------------------------------------------------*/
/* helper functions for qsort                                                 */
/*----------------------------------------------------------------------------*/
static int
iv_asc(void const * const a, void const * const b)
{
  return (((struct iv const *)a)->k < ((struct iv const *)b)->k) ? -1 : 1;
}

static int
iv_dsc(void const * const a, void const * const b)
{
  return (((struct iv const *)a)->k > ((struct iv const *)b)->k) ? -1 : 1;
}

static int
vi_asc(void const * const a, void const * const b)
{
  return (((struct vi const *)a)->k < ((struct vi const *)b)->k) ? -1 : 1;
}

static int
vi_dsc(void const * const a, void const * const b)
{
  return (((struct vi const *)a)->k > ((struct vi const *)b)->k) ? -1 : 1;
}

/*----------------------------------------------------------------------------*/
/*! Function to sort non-zero lists for each row by column id.                */
/*----------------------------------------------------------------------------*/
static int
Matrix_sort_col(Matrix * const M, int const sort)
{
  int ret = 0;

  /* unpack matrix */
  ind_t const nr = M->nr;
  ind_t const nc = M->nc;
  ind_t const * const ia = M->ia;
  ind_t       * const ja = M->ja;
  val_t       * const a  = M->a;

  /* validate input */
  if (!pp_all(nr, ia, ja))
    return -1;
  if (ASC != sort && DSC != sort)
    return -1;

#ifdef _OPENMP
# pragma omp parallel reduction(min:ret)
#endif
  {
    struct iv * kv = malloc(nc * sizeof(*kv));

    if (NULL == kv) {
      ret = -1;
      goto OMP_CLEANUP;
    }

#ifdef _OPENMP
#   pragma omp for schedule(dynamic, 8192)
#endif
    for (ind_t i = 0; i < nr; i++) {
      for (ind_t k = 0, j = ia[i]; j < ia[i + 1]; j++, k++) {
        kv[k].k = ja[j];
        if (NULL != a)
          kv[k].v = a[j];
      }

      qsort(kv, ia[i + 1] - ia[i], sizeof(*kv), ASC == sort ? iv_asc : iv_dsc);

      for (ind_t k = 0, j = ia[i]; j < ia[i + 1]; j++, k++) {
        ja[j] = kv[k].k;
        if (NULL != a)
          a[j] = kv[k].v;
      }
    }

    free(kv);

    OMP_CLEANUP:
    ;
  }

  if (0 == ret)
    M->sort = sort;

  return ret;
}

/*----------------------------------------------------------------------------*/
/*! Function to sort non-zero lists for each row by non-zero value.           */
/*----------------------------------------------------------------------------*/
static int
Matrix_sort_val(Matrix * const M, int const sort)
{
  int ret = 0;

  /* unpack matrix */
  ind_t const nr = M->nr;
  ind_t const nc = M->nc;
  ind_t const * const ia = M->ia;
  ind_t       * const ja = M->ja;
  val_t       * const a  = M->a;

  /* validate input */
  if (!pp_all(nr, nc, ia, ja, a))
    return -1;
  if (ASC != sort && DSC != sort)
    return -1;

#ifdef _OPENMP
# pragma omp parallel reduction(min:ret)
#endif
  {
    struct vi * kv = malloc(nc * sizeof(*kv));

    if (NULL == kv) {
      ret = -1;
      goto OMP_CLEANUP;
    }

#ifdef _OPENMP
#   pragma omp for schedule(dynamic, 8192)
#endif
    for (ind_t i = 0; i < nr; i++) {
      for (ind_t k = 0, j = ia[i]; j < ia[i + 1]; j++, k++) {
        kv[k].v = ja[j];
        kv[k].k = a[j];
      }

      qsort(kv, ia[i + 1] - ia[i], sizeof(*kv), ASC == sort ? vi_asc : vi_dsc);

      for (ind_t k = 0, j = ia[i]; j < ia[i + 1]; j++, k++) {
        ja[j] = kv[k].v;
        a[j] = kv[k].k;
      }
    }

    free(kv);

    OMP_CLEANUP:
    ;
  }

  if (0 == ret)
    M->sort = sort;

  return ret;
}

/*----------------------------------------------------------------------------*/
/*! Function to sort non-zero lists for each row.                             */
/*----------------------------------------------------------------------------*/
EFIKA_EXPORT int
Matrix_sort(Matrix * const M, int const flags)
{
  switch (flags & TYPE_FLAGS) {
    case COL:
      return Matrix_sort_col(M, flags & ORDER_FLAGS);
    case VAL:
      return Matrix_sort_val(M, flags & ORDER_FLAGS);
    default:
      return -1;
  }
}
