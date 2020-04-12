/* SPDX-License-Identifier: MIT */
#ifndef PP_ANY_H
#define PP_ANY_H 1

#define PP_ANY(hd, ...)               \
  ( (hd) ||                           \
  PP_IFELSE(PP_HASARGS(__VA_ARGS__))( \
    PP_DEFER2(PP__ANY)()(__VA_ARGS__) \
  )(                                  \
    (__VA_ARGS__)                     \
  ) )
#define PP__ANY() PP_ANY

/*----------------------------------------------------------------------------*/
/*! pp_any. */
/*----------------------------------------------------------------------------*/
#define pp_any(...) PP_EVAL(PP_ANY(__VA_ARGS__))

#endif /* PP_ANY_H */
