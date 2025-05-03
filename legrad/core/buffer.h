#pragma once

#include <cstdlib>
#include <functional>
#include <memory>

#include "internal/func_cmp.h"

namespace legrad::core
{
using DeleterFn = std::function<void(void*)>;
using ComparableDeleterFn = legrad::internal::function_comparable<void(void*)>;
using ContextPtr = std::unique_ptr<void, DeleterFn>;

// Do nothing for default deleter
static const DeleterFn default_deleter = [](void*) {};

class RawBuffer
{
public:
  RawBuffer()
      : ptr_(nullptr)
      , ctx_(nullptr, default_deleter)
  {
  }

  RawBuffer(void* ptr)
      : ptr_(ptr)
      , ctx_(nullptr, default_deleter)
  {
  }

  RawBuffer(void* ptr, void* ctx, DeleterFn deleter)
      : ptr_(ptr)
      , ctx_(ctx, std::move(deleter))
  {
  }

  void clear()
  {
    ptr_ = nullptr;
    // setting the unique_ptr to null will automatically delete it
    ctx_ = nullptr;
  }

  const void* ptr() const { return ptr_; }
  void* ptr() { return ptr_; }
  const void* ctx() const { return ctx_.get(); }
  void* ctx() { return ctx_.get(); }
  void* release_ctx() { return ctx_.release(); }
  ContextPtr&& move_context() { return std::move(ctx_); }
  DeleterFn get_deleter() const { return ctx_.get_deleter(); }
  operator bool() const { return ptr_ || ctx_; }

  /*
   * Instead of using `std::function` we will ComparableDeleterFn to compare two
   * function. But why we need to compare function ? We should know we don't
   * store initial DeleterFn in RawBuffer class. So if want to change the
   * DeleterFn (we can't just change this), we have to know the original
   * DeleterFn then change it with new DeleterFn.
   * Btw you should read it more in:
   * https://github.com/pytorch/pytorch/blob/v2.0.0/c10/core/Allocator.h
   */
  [[nodiscard]] bool exchange_deleter(ComparableDeleterFn expected_deleter,
                                      ComparableDeleterFn new_deleter)
  {
    // Get the current deleter as a ComparableDeleterFn for comparison
    ComparableDeleterFn current_deleter = get_deleter();

    if (current_deleter != expected_deleter)
      return false;

    // Create a new unique_ptr with the released pointer and the new deleter
    ctx_ = std::unique_ptr<void, DeleterFn>(
        ctx_.release(), static_cast<DeleterFn>(new_deleter));
    return true;
  }

  /*
   * For example, if Context is a struct, we can cast it back to struct.
   * But for the same reason as above, we need to know context exactly (which is
   * what deleter it has)
   */
  template <typename T>
  T* cast_context(ComparableDeleterFn expected_deleter) const
  {
    ComparableDeleterFn current_deleter = get_deleter();

    if (current_deleter != expected_deleter)
      return false;

    return static_cast<T*>(ctx());
  }

protected:
  void* ptr_;

  /*
   * https://github.com/pytorch/pytorch/blob/v2.0.0/c10/core/Allocator.h
   * https://github.com/pytorch/pytorch/blob/v2.0.0/c10/util/UniqueVoidPtr.h
   * Note that for the most simple case, the context will store a size of buffer
   * and buffer itself.
   * For some may confused:
   * - The context pointer will store the information on "how to deallocate data
   * pointer" correctly.
   * - So the ctx_ will store the data ptr_ too and other information.
   * - Then when we delete the RawBuffer (out of scope or something), it will
   * call the DeleterFn to delete ctx_ (which is deleting the data too)
   * - But what if the ptr_ is null ? the DeleterFn still called because ctx_ is
   * not null and it will delete something else (remember that ctx_ stores other
   * things too)
   */
  ContextPtr ctx_;
};

class Buffer
{
public:
  Buffer()
      : data_()
  {
  }

  Buffer(void* ptr)
      : data_(ptr)
  {
  }

  Buffer(void* ptr, void* ctx, DeleterFn deleter)
      : data_(ptr, ctx, deleter)
  {
  }

  RawBuffer& get_raw_data() { return data_; }
  void clear() { data_.clear(); }
  const void* get() const { return data_.ptr(); }
  void* get() { return data_.ptr(); }
  const void* get_ctx() const { return data_.ctx(); }
  void* get_ctx() { return data_.ctx(); }
  void* release_ctx() { return data_.release_ctx(); }
  operator bool() const { return static_cast<bool>(data_); }

  [[nodiscard]] bool exchange_deleter(ComparableDeleterFn expected_deleter,
                                      ComparableDeleterFn new_deleter)
  {
    return data_.exchange_deleter(expected_deleter, new_deleter);
  }

private:
  RawBuffer data_;
};
}  // namespace legrad::core