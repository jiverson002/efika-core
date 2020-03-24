/* SPDX-License-Identifier: MIT */
#ifndef PP_DEFER_H
#define PP_DEFER_H 1

#define PP_EMPTY()

#define PP_DEFER2(m) m PP_EMPTY PP_EMPTY()()
#define PP_DEFER3(m) m PP_EMPTY PP_EMPTY PP_EMPTY()()()

#endif /* PP_DEFER_H */
