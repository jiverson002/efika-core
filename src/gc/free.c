/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core/gc.h"

/*----------------------------------------------------------------------------*/
/*! Free an allocation. */
/*----------------------------------------------------------------------------*/
extern void
GC_free_impl(
  unsigned const gc_ctr,
  unsigned const gc_pctr,
  void **gc_ptr,
  void ***gc_pptr,
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
}
