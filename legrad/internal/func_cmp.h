#pragma once

#include <functional>
#include <typeindex>

namespace legrad::internal
{
/*
 * This is a way to compare two std::function in C++
 * https://stackoverflow.com/a/13105074/18845779
 */
template <typename... Args>
class function_comparable : public std::function<Args...>
{
public:
  using base_type = std::function<Args...>;

  function_comparable() = default;

  template <typename F>
  function_comparable(F&& f)
      : base_type(std::forward<F>(f))
      , type_index_(typeid_of(f))
  {
  }

  template <typename F>
  function_comparable& operator=(F&& f)
  {
    base_type::operator=(std::forward<F>(f));
    type_index_ = typeid_of(f);
    return *this;
  }

  friend bool operator==(const function_comparable& lhs,
                         const function_comparable& rhs)
  {
    // Check if both are empty
    if (!lhs && !rhs)
      return true;

    // Check if type_index_ are the same and both are not empty
    return (lhs.type_index_ == rhs.type_index_)
        && (static_cast<bool>(lhs) == static_cast<bool>(rhs));
  }

  friend bool operator!=(const function_comparable& lhs,
                         const function_comparable& rhs)
  {
    return !(lhs == rhs);
  }

  friend void swap(function_comparable& lhs, function_comparable& rhs) noexcept
  {
    using std::swap;
    swap(static_cast<base_type&>(lhs), static_cast<base_type&>(rhs));
    swap(lhs.type_index_, rhs.type_index_);
  }

private:
  template <typename F>
  static std::type_index typeid_of(const F&)
  {
    return std::type_index(typeid(typename std::decay<F>::type));
  }

  std::type_index type_index_ = std::type_index(
      typeid(void));  // Default to void type for empty function_comparable
};
}  // namespace legrad::internal