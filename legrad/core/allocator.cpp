#include <cstdlib>
#include <exception>

#include "allocator.h"
#include "core/buffer.h"
#include "macros/log.h"
#include "utils/legrad_def.h"

namespace legrad::cpu
{
void* CPUAllocator::alloc_and_throw(size_t nbytes)
{
  void* ptr = nullptr;
  // If the size is multiple of aligment default size
  // We use aligned alloc, alligned memory is sometimes
  // better than normal memory
  if (nbytes % def::MEMORY_ALIGNMENT_SIZE == 0) {
    ptr = std::aligned_alloc(def::MEMORY_ALIGNMENT_SIZE, nbytes);
  } else {
    ptr = std::malloc(nbytes);
  }
  if (ptr == nullptr) {
    LEGRAD_LOG_ERR("Cannot allocate memory with size: {}", nbytes)
    throw std::bad_alloc();
  }
  return ptr;
}

void CPUAllocator::deallocate(void* ctx)
{
  if (ctx == nullptr) {
    return;
  }

  CPUAllocator::Context* cpu_ctx = static_cast<CPUAllocator::Context*>(ctx);

  if (cpu_ctx->allocator == nullptr) {
    delete cpu_ctx;
    LEGRAD_THROW_ERROR(std::runtime_error,
                       "The context pointer has empty allocator", 0);
  }

  LEGRAD_LOG_TRACE("Delete Buffer with pointer {} and context {}", cpu_ctx->ptr,
                   fmt::ptr(cpu_ctx));
  cpu_ctx->allocator->free(cpu_ctx->ptr);
  delete cpu_ctx;
}

void CPUAllocator::free(void* ptr)
{
  std::lock_guard<std::mutex> loc(mtx_);
  std::free(ptr);
}

core::Buffer CPUAllocator::malloc(size_t nbytes)
{
  std::lock_guard<std::mutex> lock(mtx_);
  void* ptr = nullptr;
  CPUAllocator::Context* ctx = nullptr;

  if (nbytes == 0) {
    LEGRAD_LOG_WARN("Allocator create buffer with 0 size", 0);
    return core::Buffer();
  }

  LEGRAD_LOG_TRACE("Allocate new buffer with size {}", nbytes);
  try {
    ptr = alloc_and_throw(nbytes);
    ctx = new CPUAllocator::Context{ptr, nbytes, this};
  } catch (const std::exception& e) {
    LEGRAD_LOG_WARN("Cannot allocate buffer ({}), retrying. Error: {}", nbytes,
                    e.what());
    try {
      ptr = alloc_and_throw(nbytes);
      ctx = new CPUAllocator::Context{ptr, nbytes, this};
    } catch (const std::exception& retry_e) {
      LEGRAD_LOG_ERR(
          "Failed to allocate buffer ({}) even after retrying. Error: "
          "{}",
          nbytes, retry_e.what());
      return core::Buffer();
    }
  }

  return core::Buffer(ptr, ctx, CPUAllocator::deallocate);
}
}  // namespace legrad::cpu