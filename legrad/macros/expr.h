#pragma once

#include <cstddef>

/*
 * NOTE: All of this macros are essentials but I just (mindlessly) copy from
 * Pytorch c10
 */

// clang-format off
/*
 * https://gcc.gnu.org/wiki/Visibility
 */
#if defined _WIN32 || defined __CYGWIN__
  #define LEGRAD_HELPER_DLL_IMPORT __declspec(dllimport)
  #define LEGRAD_HELPER_DLL_EXPORT __declspec(dllexport)
  #define LEGRAD_HELPER_DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define LEGRAD_HELPER_DLL_IMPORT __attribute__((visibility("default")))
    #define LEGRAD_HELPER_DLL_EXPORT __attribute__((visibility("default")))
    #define LEGRAD_HELPER_DLL_LOCAL __attribute__((visibility("hidden")))
  #else
    #define LEGRAD_HELPER_DLL_IMPORT
    #define LEGRAD_HELPER_DLL_EXPORT
    #define LEGRAD_HELPER_DLL_LOCAL
  #endif
#endif

#ifdef LEGRAD_SHARED_LIB
  #ifdef LEGRAD_DLL_EXPORTS
    #define LEGRAD_API LEGRAD_HELPER_DLL_EXPORT
  #else
    #define LEGRAD_API LEGRAD_HELPER_DLL_IMPORT
  #endif
  #define LEGRAD_LOCAL LEGRAD_HELPER_DLL_LOCAL
#else
  #define LEGRAD_API
  #define LEGRAD_LOCAL
#endif

#define LEGRAD_DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete; \
  classname& operator=(const classname&) = delete

#define LEGRAD_DISABLE_MOVE_AND_ASSIGN(classname) \
  classname(classname&&) = delete; \
  classname& operator=(classname&&) = delete

#define LEGRAD_DEFAULT_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = default; \
  classname& operator=(const classname&) = default

#define LEGRAD_DEFAULT_MOVE_AND_ASSIGN(classname) \
  classname(classname&&) = default; \
  classname& operator=(classname&&) = default

/*
 * https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html
 */
#define LEGRAD_STRINGIZE_IMPL(str) #str
#define LEGRAD_STRINGIZE(str) LEGRAD_STRINGIZE_IMPL(str)

/*
 * https://gcc.gnu.org/onlinedocs/cpp/Stringizing.html
 */
#define LEGRAD_CONCAT_IMPL(s1, s2) s1##s2
#define LEGRAD_CONCAT(s1, s2) LEGRAD_CONCAT_IMPL(s1, s2)
#define LEGRAD_CONCAT_SNAKE(s1, s2) LEGRAD_CONCAT(s1, LEGRAD_CONCAT(_, s2))

/*
 * https://stackoverflow.com/questions/66593868/understanding-the-behavior-of-cs-preprocessor-when-a-macro-indirectly-expands-i
 */
#define LEGRAD_MACRO_EXPAND(args) args

#if defined(__GNUC__) || defined(__ICL) || defined(__clang__)
  #define LEGRAD_LIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 1))
  #define LEGRAD_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
  #define LEGRAD_LIKELY(expr) (expr)
  #define LEGRAD_UNLIKELY(expr) (expr)
#endif

/*
 * https://gcc.gnu.org/onlinedocs/gcc/Inline.html
 */
#if defined(_MSC_VER)
  #define LEGRAD_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
  #define LEGRAD_INLINE __attribute__((__always_inline__)) inline
#else
  #define LEGRAD_INLINE inline
#endif