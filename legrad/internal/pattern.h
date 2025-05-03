#pragma once

#include <utility>

#include "macros/expr.h"

namespace legrad::internal
{
template <typename T>
class Singleton
{
public:
  template <typename... Args>
  static T& instance(Args&&... args)
  {
    static T instance(std::forward<Args>(args)...);
    return instance;
  }

  LEGRAD_DISABLE_COPY_AND_ASSIGN(Singleton);
  LEGRAD_DISABLE_MOVE_AND_ASSIGN(Singleton);

protected:
  Singleton() = default;
  ~Singleton() = default;
};
}  // namespace legrad::internal