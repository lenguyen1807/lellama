#pragma once

#include <cstddef>

#include "core/buffer.h"

namespace legrad::core
{
class Allocator
{
public:
  virtual ~Allocator() {};
  virtual Buffer allocate(size_t) = 0;
};
}  // namespace legrad::core