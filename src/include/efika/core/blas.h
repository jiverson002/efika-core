/* SPDX-License-Identifier: MIT */
#ifndef EFIKA_CORE_BLAS_H
#define EFIKA_CORE_BLAS_H 1

#include "efika/core/rename.h"
#include "efika/core.h"

/*----------------------------------------------------------------------------*/
/*! Private API. */
/*----------------------------------------------------------------------------*/
#ifdef __cplusplus
extern "C" {
#endif

//void  efika_BLAS_vgthrz(ind_t, val_t *, val_t *, ind_t const *);

void  efika_BLAS_csrcsc(ind_t, ind_t, ind_t const *, ind_t const *,
                        val_t const *, ind_t *, ind_t *, val_t *);
val_t efika_BLAS_vdoti(ind_t, val_t const *, ind_t const *, val_t const *);
val_t efika_BLAS_vnrm2(ind_t, val_t const *);
void  efika_BLAS_vscal(ind_t, val_t, val_t *);
void  efika_BLAS_vsctr(ind_t, val_t const *, ind_t const *, val_t *);
void  efika_BLAS_vsctrz(ind_t, ind_t const *, val_t *);

#ifdef __cplusplus
}
#endif

#endif /* EFIKA_CORE_BLAS_H */
