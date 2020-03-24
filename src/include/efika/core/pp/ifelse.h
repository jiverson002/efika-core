/* SPDX-License-Identifier: MIT */
#ifndef PP_IFELSE_H
#define PP_IFELSE_H 1

#include "bool.h"

#define PP_IFELSE(condition)  PP__IFELSE(PP_BOOL(condition))
#define PP__IFELSE(condition) PP_CAT(PP_IF_, condition)

#define PP_IF_1(...) __VA_ARGS__ PP_IF_1_ELSE
#define PP_IF_0(...)             PP_IF_0_ELSE

#define PP_IF_1_ELSE(...)
#define PP_IF_0_ELSE(...) __VA_ARGS__

#endif /* PP_IFELSE_H */
