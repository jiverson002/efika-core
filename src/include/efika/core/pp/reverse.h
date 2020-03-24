/* SPDX-License-Identifier: MIT */
#ifndef PP_REVERSE_H
#define PP_REVERSE_H 1

#include "defer.h"
#include "eval.h"
#include "hasargs.h"
#include "ifelse.h"

/*----------------------------------------------------------------------------*/
/*! pp_reverse. */
/*----------------------------------------------------------------------------*/
#define PP_REVERSE(hd, ...)                 \
  PP_IFELSE(PP_HASARGS(__VA_ARGS__))(       \
     PP_DEFER2(PP__REVERSE)()( __VA_ARGS__) \
  )(                                        \
    __VA_ARGS__                             \
  ), hd
#define PP__REVERSE() PP_REVERSE

#define pp_reverse(...) PP_EVAL(PP_REVERSE(__VA_ARGS__))

#endif /* PP_REVERSE_H */
