#include <cstring>
#include <stdexcept>

#include "macros/expr.h"
#include "macros/log.h"
#include "view_pack.h"

namespace legrad::internal
{
view_pack::view_pack(const view_pack& other)
    : dim_(other.dim_)
{
  if (LEGRAD_LIKELY(other.is_inline())) {
    copy_inline_storage(other);
  } else {
    out_of_line_storage_ = allocate_new_storage(dim_);
    copy_outline_storage(other);
  }
}

view_pack& view_pack::operator=(const view_pack& other)
{
  if (this == &other) {
    return *this;
  }

  // Case 1: other is inline
  if (LEGRAD_LIKELY(other.is_inline())) {
    // free outline before copy to inlin
    if (LEGRAD_UNLIKELY(!is_inline())) {
      std::free(out_of_line_storage_);
    }
    copy_inline_storage(other);
  } else {  // Case 2: other is outline
    // resize or allocate outline storage
    // before copy the data
    if (is_inline()) {
      out_of_line_storage_ = allocate_new_storage(other.dim_);
    } else {
      resize_out_of_line_storage(other.dim_, dim_);
    }
    copy_outline_storage(other);
  }

  dim_ = other.dim_;
  return *this;
}

view_pack::view_pack(view_pack&& other) noexcept
    : dim_(other.dim_)
{
  if (LEGRAD_LIKELY(other.is_inline())) {
    copy_inline_storage(other);
    other.inline_storage_.fill(0);
  } else {
    out_of_line_storage_ = other.out_of_line_storage_;
    other.out_of_line_storage_ = nullptr;
  }
  other.dim_ = 0;
}

view_pack& view_pack::operator=(view_pack&& other) noexcept
{
  if (this == &other) {
    return *this;
  }

  if (LEGRAD_LIKELY(other.is_inline())) {
    if (LEGRAD_UNLIKELY(!is_inline())) {
      std::free(out_of_line_storage_);
    }
    copy_inline_storage(other);
    other.inline_storage_.fill(0);
  } else {
    if (is_inline()) {
      out_of_line_storage_ = allocate_new_storage(other.dim_);
    } else {
      resize_out_of_line_storage(other.dim_, dim_);
    }
    copy_outline_storage(other);
    other.out_of_line_storage_ = nullptr;
  }

  dim_ = other.dim_;
  other.dim_ = 0;
  return *this;
}

void view_pack::resize_storage(size_t new_dim)
{
  const size_t old_dim = dim_;

  if (new_dim == old_dim) {
    return;
  }

  /*
   * Resize function has two main paths for optimization:
   * - Path 1 (Fastest): Inline storage is used, and the new dimension is still
   * within the inline storage limit (<= LEGRAD_VIEW_PACK_MAX_DIM) and is
   * larger than the old dimension. In this case, it's a fast inline resize
   * operation.
   * - Path 2 (Slower): Handles all other cases, including:
   *   - Out-of-line storage resizing.
   *   - Transitions between inline and out-of-line storage.
   */

  if (LEGRAD_LIKELY(new_dim <= LEGRAD_VIEW_PACK_MAX_DIM && is_inline())) {
    if (old_dim < new_dim) {
      size_t range = new_dim - old_dim;
      std::fill_n(inline_storage_.begin() + old_dim, range, 0);
      std::fill_n(inline_storage_.begin() + LEGRAD_VIEW_PACK_MAX_DIM + old_dim,
                  range, 0);
    }
    /*
     * For the case where new_dim < old_dim (shrinking inline dimension),
     * no explicit action is needed in inline storage. We just update the
     * dimension `dim_` later. The extra data beyond the new dimension in inline
     * storage is simply ignored.
     */
    dim_ = new_dim;
    return;
  }

  /*
   * Path 2:
   * We will have 3 sub-cases within this path:
   * - Case 1: Transitioning from out-of-line to inline storage.
   *   (new_dim is always < old_dim in this case).
   * - Case 2a: Transitioning from inline to out-of-line storage.
   *   (new_dim is always > old_dim in this case).
   * - Case 2b: Already using out-of-line storage, just resizing the out-of-line
   * storage.
   */

  if (new_dim <= LEGRAD_VIEW_PACK_MAX_DIM) {
    // Case 1
    move_out_to_inline_storage(new_dim, old_dim);
  } else {
    if (is_inline()) {
      // Case 2.a
      move_inline_to_out_storage(new_dim, old_dim);
    } else {
      // Case 2.b
      resize_out_of_line_storage(new_dim, old_dim);
    }
  }
}

void view_pack::move_out_to_inline_storage(size_t new_dim, size_t old_dim)
{
  LEGRAD_ASSERT(!is_inline(), "The case in line should have been hit before",
                0);
  // We cannot use out_of_line_storage_ directly
  // because union will destroy it when
  // inline_storage_ is available
  Int* temp_storage = out_of_line_storage_;
  // copy out-of-line shape to inline
  std::copy_n(temp_storage, new_dim, inline_storage_.begin());
  // copy out-of-line stride to inline
  std::copy_n(temp_storage + old_dim, new_dim,
              inline_storage_.begin() + LEGRAD_VIEW_PACK_MAX_DIM);
  // clear out of line storage
  std::free(temp_storage);
  dim_ = new_dim;
}

void view_pack::move_inline_to_out_storage(size_t new_dim, size_t old_dim)
{
  // create new out-of-line storage
  Int* temp_storage = allocate_new_storage(new_dim);
  LEGRAD_CHECK_AND_THROW(temp_storage != nullptr, std::runtime_error,
                         "Cannot allocate new storage to change view_pack!", 0);

  size_t range = new_dim - old_dim;

  // First copy inline shape to outline shape
  std::copy_n(inline_storage_.begin(), old_dim, temp_storage);
  std::fill_n(temp_storage + old_dim, range, 0);

  // copy inline stride to outline stride
  std::copy_n(inline_storage_.begin() + LEGRAD_VIEW_PACK_MAX_DIM, old_dim,
              temp_storage + new_dim);
  std::fill_n(temp_storage + new_dim + old_dim, range, 0);

  out_of_line_storage_ = temp_storage;
  dim_ = new_dim;
}

void view_pack::resize_out_of_line_storage(size_t new_dim, size_t old_dim)
{
  const bool is_growing = new_dim > old_dim;
  if (is_growing) {
    /*
     * We need to resize the out_of_line storage
     * to have enough space
     */
    reallocate_out_of_line_storage(new_dim);
  }
  // then move stride to new starting point
  std::memmove(out_of_line_storage_ + new_dim, out_of_line_storage_ + old_dim,
               std::min(new_dim, old_dim) * sizeof(Int));
  if (!is_growing) {
    /*
     * Why we need this ?
     * Because if new dim < old dim, we need to move the data first
     * then we "strip" the data down by resizing it
     * If we resize it first, then we will lost the data
     * And after resize, we're done
     */
    reallocate_out_of_line_storage(new_dim);
  } else {
    /*
     * But for growing case
     * We need to zero the rest of shape and stride
     */
    size_t range = new_dim - old_dim;
    std::fill_n(out_of_line_storage_ + old_dim, range, 0);
    std::fill_n(out_of_line_storage_ + new_dim + old_dim, range, 0);
  }
  dim_ = new_dim;
}

Int* view_pack::allocate_new_storage(size_t n)
{
  // We need to use malloc because we will use realloc later
  Int* result = static_cast<Int*>(std::malloc(storage_bytes(n)));
  LEGRAD_CHECK_AND_THROW(result != nullptr, std::runtime_error,
                         "Cannot allocate new storage with size: {}", n);
  return result;
}

void view_pack::reallocate_out_of_line_storage(size_t n)
{
  LEGRAD_DEFAULT_ASSERT(!is_inline());
  out_of_line_storage_ =
      static_cast<Int*>(std::realloc(out_of_line_storage_, storage_bytes(n)));
  LEGRAD_CHECK_AND_THROW(
      out_of_line_storage_ != nullptr, std::runtime_error,
      "Cannot reallocate out of line storage to new size: {}", n);
}
};  // namespace legrad::internal