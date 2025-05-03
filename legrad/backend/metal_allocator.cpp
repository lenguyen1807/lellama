#include <stdexcept>

#include "core/buffer.h"
#include "macros/log.h"
#include "metal_allocator.h"
#include "utils/legrad_def.h"

namespace legrad::metal
{
MTL::Buffer* Allocator::alloc_and_throw(size_t nbytes)
{
  MTL::Buffer* ptr = device_->newBuffer(nbytes, MTL::ResourceStorageModeShared);

  if (ptr == nullptr) {
    LEGRAD_LOG_ERR("Cannot allocate buffer with size {}", nbytes);
    throw std::bad_alloc();
  }

  return ptr;
}

void Allocator::free_buffer(MTL::Buffer* buf)
{
  buf->release();
}

BucketAllocator::~BucketAllocator()
{
  LEGRAD_LOG_TRACE("Allocator destruct called", 0);
  free_cached();
}

void BucketAllocator::free_cached()
{
  std::lock_guard<std::mutex> lock(mtx_);
  for (auto& [size, ptr] : pool_) {
    LEGRAD_ASSERT(ptr != nullptr,
                  "Null buffer found in pool during free_cached", 0);
    LEGRAD_LOG_TRACE("Release buffer with pointer {} and size {}",
                     ptr->contents(), size);
    free_buffer(ptr);
  }
  pool_.clear();
}

size_t BucketAllocator::find_bucket(size_t nbytes)
{
  // First round up current size to nearest bucket size
  size_t expected_size = 0;
  for (const auto& size : def::BUCKET_SIZES) {
    if (size > nbytes) {
      expected_size = size;
      break;
    }
  }
  return expected_size;
}

core::Buffer BucketAllocator::allocate(size_t nbytes)
{
  std::lock_guard<std::mutex> lock(mtx_);

  MTL::Buffer* ptr = nullptr;
  Allocator::Context* ctx = nullptr;

  if (nbytes == 0) {
    LEGRAD_LOG_WARN("Allocator create buffer with 0 size", 0);
    // Return an empty buffer - note the context is null, deleter is default
    return core::Buffer();
  }

  size_t expected_size = find_bucket(nbytes);
  if (expected_size == 0) {
    LEGRAD_LOG_ERR(
        "The size of Buffer {} is exceeded max value of bucket which is {}. "
        "Please use another allocator or increase bucket sizes.",
        nbytes, def::BUCKET_SIZES[def::MAX_BUCKET_SIZE - 1]);
    // Consider throwing or returning an empty/error buffer
    return core::Buffer();
  }

  auto it = pool_.find(expected_size);

  if (it != pool_.end()) {
    // --- Reusing from pool ---
    ptr = it->second;
    pool_.erase(it);
    LEGRAD_ASSERT(ptr != nullptr, "Data from Allocator pool cannot be null", 0);
    LEGRAD_LOG_TRACE("Reusing buffer from pool. Bucket size: {}",
                     expected_size);
    ctx = new Allocator::Context{expected_size, nbytes, ptr, this};
  } else {
    // --- Allocating new buffer ---
    LEGRAD_LOG_TRACE("Allocating new buffer. Requested: {}, Bucket size: {}",
                     nbytes, expected_size);
    try {
      ptr = alloc_and_throw(expected_size);
      ctx = new Allocator::Context{expected_size, nbytes, ptr, this};
    } catch (const std::exception& e) {
      LEGRAD_LOG_WARN(
          "Cannot allocate buffer ({}), freeing cache and retrying. Error: {}",
          expected_size, e.what());
      free_cached();
      LEGRAD_LOG_WARN("All caches from Allocator are deleted", 0);
      // Retry allocation
      try {
        ptr = alloc_and_throw(expected_size);
        ctx = new Allocator::Context{expected_size, nbytes, ptr, this};
      } catch (const std::exception& retry_e) {
        LEGRAD_LOG_ERR(
            "Failed to allocate buffer ({}) even after freeing cache. Error: "
            "{}",
            expected_size, retry_e.what());
        return core::Buffer();
      }
    }
  }

  // Return the Buffer, passing the Metal buffer's contents pointer,
  // the context pointer (which manages the Metal buffer lifetime),
  // and the static deallocate function.
  return core::Buffer(ptr->contents(), ctx, BucketAllocator::deallocate);
}

void BucketAllocator::deallocate(void* ctx)
{
  if (ctx == nullptr) {
    return;
  }
  Allocator::Context* metal_ctx = static_cast<Allocator::Context*>(ctx);
  if (metal_ctx->allocator == nullptr) {
    // Note that we still try to delete the context of pointer
    delete metal_ctx;
    LEGRAD_THROW_ERROR(std::runtime_error,
                       "The context pointer has empty allocator", 0);
  }
  metal_ctx->allocator->return_mem(metal_ctx);
  delete metal_ctx;
}

void BucketAllocator::return_mem(Allocator::Context* ctx)
{
  std::lock_guard<std::mutex> loc(mtx_);

  if (ctx == nullptr) {
    LEGRAD_THROW_ERROR(std::invalid_argument, "return_mem called with nullptr",
                       0);
    return;
  }

  // Return memory to pool
  pool_.insert({ctx->bucket_size, ctx->buffer});
}
}  // namespace legrad::metal