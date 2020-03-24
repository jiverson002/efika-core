/* SPDX-License-Identifier: MIT */
#ifndef PP_FOLD_H
#define PP_FOLD_H 1

#include "defer.h"
#include "eval.h"
#include "ifelse.h"
#include "reverse.h"

/*----------------------------------------------------------------------------*/
/*! pp_foldr. */
/*----------------------------------------------------------------------------*/
#define PP_FOLDR(f, z, hd, ...)                \
  f ((hd),                                     \
  PP_IFELSE(PP_HASARGS(__VA_ARGS__))(          \
     PP_DEFER3(PP__FOLDR)()(f, z, __VA_ARGS__) \
  )(                                           \
    f ((__VA_ARGS__), (z))                     \
  ))
#define PP__FOLDR() PP_FOLDR

#define pp_foldr(...) PP_EVAL(PP_FOLDR(__VA_ARGS__))

/*----------------------------------------------------------------------------*/
/*! pp_foldl. */
/*----------------------------------------------------------------------------*/
#define PP__FOLDL(f, z, hd, ...)                \
  f (                                           \
  PP_IFELSE(PP_HASARGS(__VA_ARGS__))(           \
     PP_DEFER3(PP___FOLDL)()(f, z, __VA_ARGS__) \
  )(                                            \
    f ((z), (__VA_ARGS__))                      \
  ), (hd))
#define PP___FOLDL() PP__FOLDL

#define PP_FOLDL(...)       PP_EVAL(PP__FOLDL(__VA_ARGS__))
#define pp_foldl(f, z, ...) PP_FOLDL(f, z, pp_reverse(__VA_ARGS__))

#endif /* PP_FOLD_H */
