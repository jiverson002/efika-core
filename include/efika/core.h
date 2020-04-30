/* SPDX-License-Identifier: MIT */
#ifndef EFIKA_CORE_H
#define EFIKA_CORE_H 1

#include "efika/core/export.h"

#ifdef __cplusplus
# ifndef restrict
#   define undef_restrict
#   define restrict
# endif
#endif

/*----------------------------------------------------------------------------*/
/*! Configure matrix index and weight types. */
/*----------------------------------------------------------------------------*/
#ifdef EFIKA_WITH_LONG
/*! Row/column id variable type. */
typedef unsigned long EFIKA_ind_t;
/*! Value variable type. */
typedef double        EFIKA_val_t;
#else
typedef unsigned EFIKA_ind_t;
typedef float    EFIKA_val_t;
#endif

/*----------------------------------------------------------------------------*/
/*! Environment settings. */
/*----------------------------------------------------------------------------*/
EFIKA_EXPORT extern int EFIKA_debug;
EFIKA_EXPORT extern int EFIKA_verbose;

/*----------------------------------------------------------------------------*/
/*! Sparse matrix data structure */
/*----------------------------------------------------------------------------*/
typedef struct EFIKA_Matrix {
  int fmt;         /*!< metis format specifier [000] */
  int sort;        /*!< [0] - rows unsorted, 1 - rows sorted asc, 2 - rows
                        sorted dsc */
  int symm;        /*!< [0] - non-symmetric, 1 - symmetric */

  EFIKA_ind_t nr;   /*!< number of rows */
  EFIKA_ind_t nc;   /*!< number of columns */
  EFIKA_ind_t nnz;  /*!< number of non-zeros */
  EFIKA_ind_t   * restrict ia; /*!< sparse matrix row index array */
  EFIKA_ind_t   * restrict ja; /*!< sparse matrix column index array */
  unsigned char * restrict za; /*!< sparse matrix z-index array */
  EFIKA_val_t   * restrict a;  /*!< sparse matrix non-zero entries */

  /*! number of weights associated with each vertex (metis only) */
  EFIKA_ind_t ncon;
  /*! vertex size (metis only) */
  EFIKA_ind_t * restrict vsiz;
  /*! vertex weights (metis only) */
  EFIKA_val_t * restrict vwgt;

  void * pp;              /*!< pointer to preprocessed data */
  void (*pp_free)(void*); /*!< pointer to function to free preprocessed data */
} EFIKA_Matrix;

/*----------------------------------------------------------------------------*/
/*! Various flags. */
/*----------------------------------------------------------------------------*/
enum EFIKA_Flag {
  EFIKA_NONE = 0x0,
  EFIKA_ASC  = 0x1,
  EFIKA_DSC  = 0x10,

  EFIKA_BFT  = 0x100,
  EFIKA_COL  = 0x1000,
  EFIKA_DEG  = 0x10000,
  EFIKA_PFX  = 0x100000,
  EFIKA_VAL  = 0x1000000,
};

/*----------------------------------------------------------------------------*/
/*! Public API. */
/*----------------------------------------------------------------------------*/
#ifdef __cplusplus
extern "C" {
#endif

EFIKA_EXPORT int  EFIKA_Matrix_comp(EFIKA_Matrix *);
EFIKA_EXPORT int  EFIKA_Matrix_copy(EFIKA_Matrix const *, EFIKA_Matrix *);
EFIKA_EXPORT int  EFIKA_Matrix_cord(EFIKA_Matrix *, int);
EFIKA_EXPORT void EFIKA_Matrix_free(EFIKA_Matrix *);
EFIKA_EXPORT int  EFIKA_Matrix_iidx(EFIKA_Matrix const *, EFIKA_Matrix *);
EFIKA_EXPORT int  EFIKA_Matrix_init(EFIKA_Matrix *);
EFIKA_EXPORT int  EFIKA_Matrix_norm(EFIKA_Matrix *);
EFIKA_EXPORT int  EFIKA_Matrix_perm(EFIKA_Matrix *, EFIKA_ind_t *, EFIKA_ind_t *);
EFIKA_EXPORT int  EFIKA_Matrix_rord(EFIKA_Matrix *, int);
EFIKA_EXPORT int  EFIKA_Matrix_sort(EFIKA_Matrix *, int);
EFIKA_EXPORT int  EFIKA_Matrix_test(EFIKA_Matrix const *);
EFIKA_EXPORT int  EFIKA_Matrix_rord(EFIKA_Matrix *, int);
EFIKA_EXPORT int  EFIKA_Matrix_sort(EFIKA_Matrix *, int);
EFIKA_EXPORT int  EFIKA_Matrix_test(EFIKA_Matrix const *);
EFIKA_EXPORT int  EFIKA_Matrix_zord(EFIKA_Matrix const *, EFIKA_Matrix *);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
# ifdef undef_restrict
#   undef restrict
# endif
#endif

#endif /* EFIKA_CORE_H */
