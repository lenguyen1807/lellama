#pragma once

#include <cstddef>
#include <mutex>

#include "core/buffer.h"

namespace legrad::core
{
class Allocator
{
public:
  virtual ~Allocator() {};
  virtual Buffer malloc(size_t) = 0;
  virtual void free(void*) = 0;
};

}  // namespace legrad::core

namespace legrad::cpu
{
/*
 * This is allocator for CPU
 */
class CPUAllocator : public core::Allocator
{
public:
  struct Context
  {
    void* ptr;
    size_t size;
    // We want to know the allocator that allocate the buffer
    CPUAllocator* allocator;
  };

  CPUAllocator() = default;
  ~CPUAllocator() {}

  core::Buffer malloc(size_t) override;
  void free(void*) override;

  static void deallocate(void*);

private:
  void* alloc_and_throw(size_t);

private:
  std::mutex mtx_;
};
}  // namespace legrad::cpu