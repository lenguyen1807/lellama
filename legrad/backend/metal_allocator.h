#pragma once

#include <cstddef>
#include <map>
#include <mutex>

#include "Metal/MTLBuffer.hpp"
#include "Metal/MTLDevice.hpp"
#include "core/allocator.h"
#include "core/buffer.h"

namespace legrad::metal
{
class Allocator : public core::Allocator
{
public:
  struct Context
  {
    size_t bucket_size;
    size_t real_size;
    MTL::Buffer* buffer;
    // We want to know the allocator that allocate this buffer
    metal::Allocator* allocator;
  };

  Allocator(MTL::Device* device)
      : core::Allocator()
      , device_(device)
  {
  }

  virtual ~Allocator() {}
  virtual void return_mem(Allocator::Context*) = 0;

protected:
  MTL::Buffer* alloc_and_throw(size_t nbytes);
  void free_buffer(MTL::Buffer* buf);

protected:
  MTL::Device* device_;
};

class BucketAllocator : public metal::Allocator
{
public:
  BucketAllocator(MTL::Device* device)
      : metal::Allocator(device)
  {
  }

  ~BucketAllocator();

  core::Buffer allocate(size_t bytes) override;
  static void deallocate(void*);
  void return_mem(Allocator::Context*) override;

private:
  void free_cached();
  size_t find_bucket(size_t nbytes);

private:
  std::mutex mtx_;
  std::multimap<size_t, MTL::Buffer*> pool_;
};
};  // namespace legrad::metal