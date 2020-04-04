/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core/gc.h"

/*----------------------------------------------------------------------------*/
/*! Free all recorded allocations. */
/*----------------------------------------------------------------------------*/
void
GC_cleanup_impl(
  unsigned const gc_ctr,
  unsigned const gc_pctr,
  unsigned const gc_fctr,
  void **gc_ptr,
  void ***gc_pptr,
  void (**gc_free)(void*),
  void **gc_fptr
)
{
  for (unsigned i = 0; i < gc_ctr; i++) {
    if (NULL != gc_ptr[i]) {
      free(gc_ptr[i]);
    }
  }

  for (unsigned i = 0; i < gc_pctr; i++) {
    if (NULL != *(gc_pptr[i])) {
      free(*(gc_pptr[i]));
    }
  }

  for (unsigned i = 0; i < gc_fctr; i++) {
    if (NULL != gc_fptr[i]) {
      gc_free[i](gc_fptr[i]);
    }
  }
}
