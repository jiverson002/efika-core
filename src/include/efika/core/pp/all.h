/* SPDX-License-Identifier: MIT */
#ifndef PP_ALL_H
#define PP_ALL_H 1

#define PP_ALL(hd, ...)               \
  ( (hd) &&                           \
  PP_IFELSE(PP_HASARGS(__VA_ARGS__))( \
    PP_DEFER2(PP__ALL)()(__VA_ARGS__) \
  )(                                  \
    (__VA_ARGS__)                     \
  ) )
#define PP__ALL() PP_ALL

/*----------------------------------------------------------------------------*/
/*! pp_all. */
/*----------------------------------------------------------------------------*/
#define pp_all(...) PP_EVAL(PP_ALL(__VA_ARGS__))

#endif /* PP_ALL_H */
