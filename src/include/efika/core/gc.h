/* SPDX-License-Identifier: MIT */
#ifndef EFIKA_CORE_GC_H
#define EFIKA_CORE_GC_H 1

#include <stdlib.h>

#include "efika/core.h"

#include "efika/core/rename.h"

/*----------------------------------------------------------------------------*/
/*! Memory allocation utilities -- not thread safe. */
/*----------------------------------------------------------------------------*/
#define GC_func_init()\
  unsigned __gc_ret = 0;\
  unsigned __gc_ctr = 0;\
  unsigned __gc_pctr = 0;\
  unsigned __gc_fctr = 0;\
  void *__gc_ptr[128];\
  void **__gc_pptr[128];\
  void (*__gc_free[128])(void*);\
  void *__gc_fptr[128]

#define GC_assert(cond)\
  if (!(cond)) {\
    GC_cleanup();\
    return -1;\
  } else (void)0

#define GC_alloc(call)\
  ( __gc_ptr[__gc_ctr] = call\
  , __gc_ptr[__gc_ctr] ? 0 : (GC_cleanup(), __gc_ret = 1)\
  , __gc_ptr[__gc_ctr++]\
  );\
  if (__gc_ret)\
    return -1;\
  else\
    (void)0

#define GC_cleanup()\
  GC_cleanup_impl(__gc_ctr, __gc_pctr, __gc_fctr, __gc_ptr, __gc_pptr,\
                  __gc_free, __gc_fptr)

#define GC_free(xptr)\
  GC_free_impl(__gc_ctr, __gc_pctr, __gc_fctr, __gc_ptr, __gc_pptr, __gc_free,\
               __gc_fptr, xptr)

#define GC_return         return GC_cleanup(),
#define GC_register(pptr) __gc_pptr[__gc_pctr++] = (void*)(pptr)
#define GC_register_free(free, fptr)\
  __gc_free[__gc_fctr++] = (void (*)(void*))free;\
  __gc_fptr[__gc_fctr] = fptr

#define GC_malloc(n)     GC_alloc(malloc(n))
#define GC_calloc(n, sz) GC_alloc(calloc(n, sz))
#define GC_realloc(xptr, n)\
  GC_realloc_impl(__gc_ctr, __gc_pctr, __gc_ptr, __gc_pptr, xptr, n)

/*----------------------------------------------------------------------------*/
/*! Bypass the garbage collector. */
/*----------------------------------------------------------------------------*/
#define unsafe_malloc  malloc
#define unsafe_calloc  calloc
#define unsafe_realloc realloc
#define unsafe_free    free

/*----------------------------------------------------------------------------*/
/*! Garbage collection. */
/*----------------------------------------------------------------------------*/
#define GC_cleanup_impl efika_GC_cleanup_impl
#define GC_free_impl    efika_GC_free_impl
#define GC_realloc_impl efika_GC_realloc_impl

/*----------------------------------------------------------------------------*/
/*! Private API. */
/*----------------------------------------------------------------------------*/
void  GC_cleanup_impl(unsigned, unsigned, unsigned, void **, void ***, void (**)(void*), void **);
void  GC_free_impl(unsigned, unsigned, unsigned, void **, void ***, void (**)(void*), void **, void *);
void* GC_realloc_impl(unsigned, unsigned, void **, void ***, void *, size_t);

#endif /* EFIKA_CORE_GC_H */
