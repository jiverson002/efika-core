/* SPDX-License-Identifier: MIT */
#ifndef PP_BOOL_H
#define PP_BOOL_H 1

#include "cat.h"

#define PP_SND(a, b, ...) b

#define PP_IS_PROBE(...) PP_SND(__VA_ARGS__, 0, XXX)
#define PP_PROBE() ~, 1

#define PP_NOT(x) PP_IS_PROBE(PP_CAT(PP_NOT_, x))
#define PP_NOT_0  PP_PROBE()

#define PP_BOOL(x) PP_NOT(PP_NOT(x))

#endif /* PP_BOOL_H */
