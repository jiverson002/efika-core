/* SPDX-License-Identifier: MIT */
#include <stdlib.h>

#include "efika/core/gc.h"
#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/*! Reallocate an allocation. */
/*----------------------------------------------------------------------------*/
void*
GC_realloc_impl(
  unsigned const gc_ctr,
  unsigned const gc_pctr,
  void **gc_ptr,
  void ***gc_pptr,
  void *ptr,
  size_t const n
)
{
  if (NULL == ptr)
    return NULL;

  for (unsigned i = 0; i < gc_ctr; i++)
    if (ptr == gc_ptr[i])
      return gc_ptr[i] = realloc(ptr, n);

  for (unsigned i = 0; i < gc_pctr; i++)
    if (ptr == *(gc_pptr[i]))
      return gc_pptr[i] = realloc(ptr, n);

  return NULL;
}
