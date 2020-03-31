/* SPDX-License-Identifier: MIT */
#ifndef EFIKA_CORE_RENAME_H
#define EFIKA_CORE_RENAME_H 1

#define ind_t  EFIKA_ind_t
#define val_t  EFIKA_val_t
#define Matrix EFIKA_Matrix

/*----------------------------------------------------------------------------*/
/*! Configure matrix index and weight types */
/*----------------------------------------------------------------------------*/
#include <inttypes.h>
#ifdef WITH_WIDE
# define IND_T    "%"PRIu64
# define VAL_T    "%lf"
# define STRTOI   strtoul
# define STRTOV   strtod
# define IND_MAX  UINT64_MAX
# define IND_MIN  0
# define VAL_MAX  DBL_MAX
# define VAL_MIN  -(DBL_MAX)
#else
# define IND_T    "%"PRIu32
# define VAL_T    "%f"
# define STRTOI   (ind_t)strtoul
# define STRTOV   (val_t)strtod
# define IND_MAX  UINT32_MAX
# define IND_MIN  0
# define VAL_MAX  FLT_MAX
# define VAL_MIN  -(FLT_MAX)
#endif

#define NONE EFIKA_NONE
#define ASC  EFIKA_ASC
#define DSC  EFIKA_DSC
#define BFT  EFIKA_BFT
#define COL  EFIKA_COL
#define DEG  EFIKA_DEG
#define PFX  EFIKA_PFX
#define VAL  EFIKA_VAL

#define strtoi(hd, tl) STRTOI(hd, tl, 0)
#define strtov         STRTOV

/*----------------------------------------------------------------------------*/
/*! Format checking macros. */
/*----------------------------------------------------------------------------*/
#define has_adjwgt(FMT) (  1 == (FMT) ||  11 == (FMT) || 101 == (FMT) || \
                         111 == (FMT))
#define has_vtxsiz(FMT) (100 == (FMT) || 101 == (FMT) || 110 == (FMT) || \
                         111 == (FMT))
#define has_vtxwgt(FMT) ( 10 == (FMT) ||  11 == (FMT) || 110 == (FMT) || \
                         111 == (FMT))

/*----------------------------------------------------------------------------*/
/*! Garbage collection. */
/*----------------------------------------------------------------------------*/
#define GC_cleanup_impl efika_GC_cleanup_impl
#define GC_free_impl    efika_GC_free_impl
#define GC_realloc_impl efika_GC_realloc_impl

#endif /* EFIKA_IMPL_RENAME_H */
