/* SPDX-License-Identifier: MIT */
#ifndef PP_MAP_H
#define PP_MAP_H 1

#include "defer.h"
#include "eval.h"
#include "ifelse.h"

#define PP_MAP(m, hd, ...)               \
  m(hd),                                 \
  PP_IFELSE(PP_HASARGS(__VA_ARGS__))(    \
    PP_DEFER2(PP__MAP)()(m, __VA_ARGS__) \
  )(                                     \
    m(__VA_ARGS__)                       \
  )
#define PP__MAP() PP_MAP

/*----------------------------------------------------------------------------*/
/*! pp_map. */
/*----------------------------------------------------------------------------*/
#define pp_map(...) PP_EVAL(PP_MAP(__VA_ARGS__))

#endif /* PP_MAP_H */
