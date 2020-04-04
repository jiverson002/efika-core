/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core/gc.h"

/*----------------------------------------------------------------------------*/
/*! Free an allocation. */
/*----------------------------------------------------------------------------*/
void
GC_free_impl(
  unsigned const gc_ctr,
  unsigned const gc_pctr,
  unsigned const gc_fctr,
  void **gc_ptr,
  void ***gc_pptr,
  void (**gc_free)(void*),
  void **gc_fptr,
  void *ptr
)
{
  if (NULL == ptr)
    return;

  for (unsigned i = 0; i < gc_ctr; i++) {
    if (ptr == gc_ptr[i]) {
      free(gc_ptr[i]);
      gc_ptr[i] = NULL;
      return;
    }
  }

  for (unsigned i = 0; i < gc_pctr; i++) {
    if (ptr == *(gc_pptr[i])) {
      free(*(gc_pptr[i]));
      gc_pptr[i] = NULL;
      return;
    }
  }

  for (unsigned i = 0; i < gc_fctr; i++) {
    if (ptr == gc_fptr[i]) {
      gc_free[i](gc_fptr[i]);
      gc_fptr[i] = NULL;
      return;
    }
  }
}
