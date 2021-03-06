/* SPDX-License-Identifier: MIT */
#ifndef EFIKA_CORE_BLAS_H
#define EFIKA_CORE_BLAS_H 1

#include "efika/core.h"

#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/*! BLAS routines. */
/*----------------------------------------------------------------------------*/
#define BLAS_csrcsc efika_BLAS_csrcsc
#define BLAS_vdoti  efika_BLAS_vdoti
#define BLAS_vnrm2  efika_BLAS_vnrm2
#define BLAS_vscal  efika_BLAS_vscal
#define BLAS_vsctr  efika_BLAS_vsctr
#define BLAS_vsctrz efika_BLAS_vsctrz

/*----------------------------------------------------------------------------*/
/*! Private API. */
/*----------------------------------------------------------------------------*/
#ifdef __cplusplus
extern "C" {
#endif

void  BLAS_csrcsc(ind_t, ind_t, ind_t const *, ind_t const *,
                  val_t const *, ind_t *, ind_t *, val_t *);
val_t BLAS_vdoti(ind_t, val_t const *, ind_t const *, val_t const *);
val_t BLAS_vnrm2(ind_t, val_t const *);
void  BLAS_vscal(ind_t, val_t, val_t *);
void  BLAS_vsctr(ind_t, val_t const *, ind_t const *, val_t *);
void  BLAS_vsctrz(ind_t, ind_t const *, val_t *);

#ifdef __cplusplus
}
#endif

#endif /* EFIKA_CORE_BLAS_H */
