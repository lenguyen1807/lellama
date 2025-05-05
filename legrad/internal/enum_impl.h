#pragma once

#include <type_traits>

#include "boost/preprocessor.hpp"
#include "macros/expr.h"
#include "macros/log.h"

/*
 * HACK: This is a hack to convert Enum to String
 * Reference:
 * https://belaycpp.com/2021/08/24/best-ways-to-convert-an-enum-to-a-string/
 * https://stackoverflow.com/questions/8357240/how-to-automatically-convert-strongly-typed-enum-into-int
 */
template <typename EnumType>
using RawEnumType = std::underlying_type_t<EnumType>;

template <typename EnumType>
constexpr auto RawEnumVal(EnumType enm) noexcept
{
  return static_cast<RawEnumType<EnumType>>(enm);
}

template <typename EnumType>
constexpr auto ToIntEnum(EnumType em) noexcept
{
  return static_cast<int>(RawEnumVal(em));
}

#define PROCESS_ONE_ELEMENT(r, unused, idx, elem) \
  BOOST_PP_COMMA_IF(idx) BOOST_PP_STRINGIZE(elem)

#define LEGRAD_ENUM_INTERNAL(name, type, ...) \
  enum class name : type \
  { \
    __VA_ARGS__ \
  }; \
  constexpr const char* LEGRAD_CONCAT(name, Strings)[] = { \
      BOOST_PP_SEQ_FOR_EACH_I(PROCESS_ONE_ELEMENT, % %, \
                              BOOST_PP_VARIADIC_TO_SEQ(__VA_ARGS__))}; \
  inline const char* LEGRAD_CONCAT(name, ToString)(name value) \
  { \
    return LEGRAD_CONCAT(name, Strings)[ToIntEnum(value)]; \
  }

namespace legrad::internal
{
/*
 * We use enum_impl to provide an iterator for enum
 * Every Enum we used should be wrapped by enum_impl
 */
template <typename ENUM_TYPE, ENUM_TYPE beginVal, ENUM_TYPE endVal>
class EnumIterator
{
  LEGRAD_STATIC_ASSERT(
      RawEnumVal(beginVal) <= RawEnumVal(endVal),
      "Cannot create iterator where 'beginVal' comes after 'endVal'!");

public:
  EnumIterator()
      : current_val_(beginVal)
  {
  }

  EnumIterator(ENUM_TYPE val)
      : current_val_(val)
  {
  }

  EnumIterator begin() const { return *this; }
  EnumIterator end() const
  {
    EnumIterator iter;
    iter.current_val_ = static_cast<ENUM_TYPE>(RawEnumVal(endVal) + 1);
    return iter;
  }

  EnumIterator& operator++()
  {  // Prefix increment
    current_val_ = static_cast<ENUM_TYPE>(RawEnumVal(current_val_) + 1);
    return *this;
  }

  EnumIterator operator++(int)
  {  // Postfix increment
    EnumIterator temp = *this;
    ++(*this);
    return temp;
  }

  ENUM_TYPE operator*() const { return current_val_; }

  bool operator!=(const EnumIterator& other) const
  {
    return RawEnumVal(current_val_) != RawEnumVal(other.current_val_);
  }

  bool operator==(const EnumIterator& other) const
  {
    return RawEnumVal(current_val_) == RawEnumVal(other.current_val_);
  }

private:
  ENUM_TYPE current_val_;
};

/*
 * If you want to define Enum, use this, this will provide you enum and its
 * iterator. You also have to provide begin value and end value of enum
 */
#define LEGRAD_ENUM(name, type, beginVal, endVal, ...) \
  LEGRAD_ENUM_INTERNAL(name, type, __VA_ARGS__) \
  using LEGRAD_CONCAT(name, Iter) = \
      legrad::internal::EnumIterator<name, name::beginVal, name::endVal>;
}  // namespace legrad::internal