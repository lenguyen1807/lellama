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
class MetalAllocator : public core::Allocator
{
public:
  struct Context
  {
    size_t bucket_size;
    size_t real_size;
    MTL::Buffer* buffer;
    // We want to know the allocator that allocate this buffer
    metal::MetalAllocator* allocator;
  };

  MetalAllocator(MTL::Device* device)
      : core::Allocator()
      , device_(device)
  {
  }

  virtual ~MetalAllocator() {}

protected:
  MTL::Buffer* alloc_and_throw(size_t nbytes);

protected:
  MTL::Device* device_;
};

class MetalBucketAllocator : public metal::MetalAllocator
{
public:
  MetalBucketAllocator(MTL::Device* device)
      : metal::MetalAllocator(device)
  {
  }

  ~MetalBucketAllocator();

  core::Buffer malloc(size_t bytes) override;
  static void deallocate(void*);
  void free(void*) override;

private:
  void free_cached();
  size_t find_bucket(size_t nbytes);

private:
  std::mutex mtx_;
  std::multimap<size_t, MTL::Buffer*> pool_;
};
};  // namespace legrad::metal