/* SPDX-License-Identifier: MIT */
#ifndef EFIKA_CORE_RENAME_H
#define EFIKA_CORE_RENAME_H 1

/*----------------------------------------------------------------------------*/
/*! Core datatypes. */
/*----------------------------------------------------------------------------*/
#define ind_t  EFIKA_ind_t
#define val_t  EFIKA_val_t
#define Matrix EFIKA_Matrix

/*----------------------------------------------------------------------------*/
/*! Global config variables. */
/*----------------------------------------------------------------------------*/
#define debug   EFIKA_debug
#define verbose EFIKA_verbose

/*----------------------------------------------------------------------------*/
/*! Core functions. */
/*----------------------------------------------------------------------------*/
#define Matrix_comp EFIKA_Matrix_comp
#define Matrix_copy EFIKA_Matrix_copy
#define Matrix_cord EFIKA_Matrix_cord
#define Matrix_free EFIKA_Matrix_free
#define Matrix_iidx EFIKA_Matrix_iidx
#define Matrix_init EFIKA_Matrix_init
#define Matrix_norm EFIKA_Matrix_norm
#define Matrix_perm EFIKA_Matrix_perm
#define Matrix_rord EFIKA_Matrix_rord
#define Matrix_rsb  EFIKA_Matrix_rsb
#define Matrix_sort EFIKA_Matrix_sort
#define Matrix_test EFIKA_Matrix_test

/*----------------------------------------------------------------------------*/
/*! Configure matrix index and weight types */
/*----------------------------------------------------------------------------*/
#include <float.h>
#include <inttypes.h>
#ifdef EFIKA_WITH_LONG
# define PRIind   "%lu"
# define PRIval   "%lf"
# define IND_MAX  ULONG_MAX
# define IND_MIN  0
# define VAL_MAX  DBL_MAX
# define VAL_MIN  -(DBL_MAX)
# define strtoi(hd, tl) (ind_t)strtoul(hd, tl, 0)
# define strtov         strtod
# define sqrtv          sqrt
#else
# define PRIind   "%u"
# define PRIval   "%f"
# define IND_MAX  UINT_MAX
# define IND_MIN  0
# define VAL_MAX  FLT_MAX
# define VAL_MIN  -(FLT_MAX)
# define strtoi(hd, tl) (ind_t)strtoul(hd, tl, 0)
# define strtov         (val_t)strtod
# define sqrtv          sqrtf
#endif

/*----------------------------------------------------------------------------*/
/*! Flags. */
/*----------------------------------------------------------------------------*/
#define NONE EFIKA_NONE
#define ASC  EFIKA_ASC
#define DSC  EFIKA_DSC
#define BFT  EFIKA_BFT
#define COL  EFIKA_COL
#define DEG  EFIKA_DEG
#define PFX  EFIKA_PFX
#define VAL  EFIKA_VAL

#define ORDER_FLAGS (ASC|DSC)
#define TYPE_FLAGS  (BFT|COL|DEG|PFX|VAL)

/*----------------------------------------------------------------------------*/
/*! Format checking macros. */
/*----------------------------------------------------------------------------*/
#define has_adjwgt(FMT) (  1 == (FMT) ||  11 == (FMT) || 101 == (FMT) || \
                         111 == (FMT))
#define has_vtxsiz(FMT) (100 == (FMT) || 101 == (FMT) || 110 == (FMT) || \
                         111 == (FMT))
#define has_vtxwgt(FMT) ( 10 == (FMT) ||  11 == (FMT) || 110 == (FMT) || \
                         111 == (FMT))

#endif /* EFIKA_CORE_RENAME_H */
