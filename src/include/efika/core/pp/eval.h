/* SPDX-License-Identifier: MIT */
#ifndef PP_EVAL_H
#define PP_EVAL_H 1

#define PP_EVAL(...)     PP_EVAL1024(__VA_ARGS__)
#define PP_EVAL1024(...) PP_EVAL512(PP_EVAL512(__VA_ARGS__))
#define PP_EVAL512(...)  PP_EVAL256(PP_EVAL256(__VA_ARGS__))
#define PP_EVAL256(...)  PP_EVAL128(PP_EVAL128(__VA_ARGS__))
#define PP_EVAL128(...)  PP_EVAL64(PP_EVAL64(__VA_ARGS__))
#define PP_EVAL64(...)   PP_EVAL32(PP_EVAL32(__VA_ARGS__))
#define PP_EVAL32(...)   PP_EVAL16(PP_EVAL16(__VA_ARGS__))
#define PP_EVAL16(...)   PP_EVAL8(PP_EVAL8(__VA_ARGS__))
#define PP_EVAL8(...)    PP_EVAL4(PP_EVAL4(__VA_ARGS__))
#define PP_EVAL4(...)    PP_EVAL2(PP_EVAL2(__VA_ARGS__))
#define PP_EVAL2(...)    PP_EVAL1(PP_EVAL1(__VA_ARGS__))
#define PP_EVAL1(...)    __VA_ARGS__

#endif /* PP_EVAL_H */
