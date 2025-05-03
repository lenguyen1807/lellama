#pragma once

#ifdef LEGRAD_USE_BOOST
#include <boost/core/null_deleter.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/attributes/timer.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/text_file_backend.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/file.hpp>
#endif

#include <cassert>

#define FMT_HEADER_ONLY
#include <fmt/format.h>

#include "expr.h"

// https://fmt.dev/11.1/api/#compile-time-checks
LEGRAD_INLINE void vlog(const char* type,
                        const char* file,
                        int line,
                        fmt::string_view fmt,
                        fmt::format_args args)
{
  fmt::print("[{}] ({}, {}): {}\n", type, file, line, fmt::vformat(fmt, args));
}

template <typename... T>
void log(const char* type,
         const char* file,
         int line,
         fmt::format_string<T...> fmt,
         T&&... args)
{
  vlog(type, file, line, fmt, fmt::make_format_args(args...));
}

// clang-format off
#ifdef LEGRAD_USE_BOOST
  #define LOG_INFO_STREAM BOOST_LOG_TRIVIAL(info)
  #define LOG_DEBUG_STREAM BOOST_LOG_TRIVIAL(debug)
  #define LOG_TRACE_STREAM BOOST_LOG_TRIVIAL(trace)
  #define LOG_ERROR_STREAM BOOST_LOG_TRIVIAL(error)
  #define LOG_WARN_STREAM BOOST_LOG_TRIVIAL(warning)
#else
  #define LOG_INFO_STREAM "info"
  #define LOG_DEBUG_STREAM "debug"
  #define LOG_TRACE_STREAM "trace"
  #define LOG_ERROR_STREAM "error"
  #define LOG_WARN_STREAM "warning"
#endif

/*
 * Not that fmt::format won't compile if __VA_ARGS__ is empty
 * The solution now is pass anything you want (0 in my case)
 * to avoid this situation
 */
 #define LEGRAD_FORMAT_STR(msg, ...) fmt::format(msg, __VA_ARGS__)

 /*
 * HACK: only get file name instead of full path
 * https://stackoverflow.com/questions/8487986/file-macro-shows-full-path
 */
#define LEGRAD_FILENAME \
(strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#ifdef USE_BOOST
#define LEGRAD_LOG_TEMPLATE(logger, msg, ...) \
logger << "(" << __PRETTY_FUNCTION__ << ", " << LEGRAD_FILENAME << ", " \
       << static_cast<uint32_t>(__LINE__) \
       << "): " << LEGRAD_FORMAT_STR(msg, __VA_ARGS__)
#else
#define LEGRAD_LOG_TEMPLATE(logger, msg, ...) log(logger, __PRETTY_FUNCTION__, __LINE__, msg, __VA_ARGS__)
#endif

#if LEGRAD_DEBUG_1
  #define LEGRAD_LOG_INFO(msg, ...) LEGRAD_LOG_TEMPLATE(LOG_INFO_STREAM, msg, __VA_ARGS__);
#else
  #define LEGRAD_LOG_INFO(msg, ...)
#endif 

#if LEGRAD_DEBUG_2
  #define LEGRAD_LOG_DEBUG(msg, ...) LEGRAD_LOG_TEMPLATE(LOG_DEBUG_STREAM, msg, __VA_ARGS__);
#else
  #define LEGRAD_LOG_DEBUG(msg, ...)
#endif

#if LEGRAD_DEBUG_3
  #define LEGRAD_LOG_TRACE(msg, ...) LEGRAD_LOG_TEMPLATE(LOG_TRACE_STREAM, msg, __VA_ARGS__);
#else
  #define LEGRAD_LOG_TRACE(msg, ...)
#endif

#define LEGRAD_LOG_ERR(msg, ...) LEGRAD_LOG_TEMPLATE(LOG_ERROR_STREAM, msg, __VA_ARGS__);
#define LEGRAD_LOG_WARN(msg, ...) LEGRAD_LOG_TEMPLATE(LOG_WARN_STREAM, msg, __VA_ARGS__);

// clang-format on
#define LEGRAD_THROW_ERROR(throw_type, msg, ...) \
  do { \
    LEGRAD_LOG_ERR(msg, __VA_ARGS__) \
    throw throw_type(LEGRAD_FORMAT_STR(msg, __VA_ARGS__)); \
  } while (false)

#define LEGRAD_STATIC_ASSERT(cond, msg) static_assert(cond, msg)

#ifdef NDEBUG
#define LEGRAD_ASSERT(cond, msg, ...) (void)(cond)
#define LEGRAD_DEFAULT_ASSERT(cond) (void)(cond)
#else
#define LEGRAD_ASSERT(cond, msg, ...) \
  do { \
    if (LEGRAD_UNLIKELY(!(cond))) { \
      LEGRAD_LOG_ERR(msg, __VA_ARGS__) \
      assert(cond); \
    } \
  } while (false)
#define LEGRAD_DEFAULT_ASSERT(cond) assert(cond);
#endif

#define LEGRAD_CHECK_AND_THROW(cond, throw_type, msg, ...) \
  do { \
    if (LEGRAD_UNLIKELY(!(cond))) { \
      LEGRAD_THROW_ERROR(throw_type, msg, __VA_ARGS__); \
    } \
  } while (false)