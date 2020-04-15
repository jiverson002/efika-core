/* SPDX-License-Identifier: MIT */
#ifndef EFIKA_CORE_H
#define EFIKA_CORE_H 1

/*----------------------------------------------------------------------------*/
/*! Configure matrix index and weight types. */
/*----------------------------------------------------------------------------*/
#ifndef EFIKA_WITH_SHORT
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
extern int EFIKA_debug;
extern int EFIKA_verbose;

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
  EFIKA_ind_t * ia; /*!< sparse matrix row index array */
  EFIKA_ind_t * ja; /*!< sparse matrix column index array */
  EFIKA_val_t * a;  /*!< sparse matrix non-zero entries */

  /*! number of weights associated with each vertex (metis only) */
  EFIKA_ind_t ncon;
  /*! vertex size (metis only) */
  EFIKA_ind_t * vsiz;
  /*! vertex weights (metis only) */
  EFIKA_val_t * vwgt;

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

int  EFIKA_Matrix_comp(EFIKA_Matrix *);
int  EFIKA_Matrix_cord(EFIKA_Matrix *, int);
void EFIKA_Matrix_free(EFIKA_Matrix *);
int  EFIKA_Matrix_iidx(EFIKA_Matrix const *, EFIKA_Matrix *);
int  EFIKA_Matrix_init(EFIKA_Matrix *);
int  EFIKA_Matrix_norm(EFIKA_Matrix *);
int  EFIKA_Matrix_perm(EFIKA_Matrix *, EFIKA_ind_t *, EFIKA_ind_t *);
int  EFIKA_Matrix_rord(EFIKA_Matrix *, int);
int  EFIKA_Matrix_sort(EFIKA_Matrix *, int);
int  EFIKA_Matrix_test(EFIKA_Matrix const *);

#ifdef __cplusplus
}
#endif

#endif /* EFIKA_CORE_H */
