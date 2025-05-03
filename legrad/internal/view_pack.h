/* From PyTorch:

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

From Caffe2:

Copyright (c) 2016-present, Facebook Inc. All rights reserved.

All contributions by Facebook:
Copyright (c) 2016 Facebook Inc.

All contributions by Google:
Copyright (c) 2015 Google Inc.
All rights reserved.

All contributions by Yangqing Jia:
Copyright (c) 2015 Yangqing Jia
All rights reserved.

All contributions by Kakao Brain:
Copyright 2019-2020 Kakao Brain

All contributions by Cruise LLC:
Copyright (c) 2022 Cruise LLC.
All rights reserved.

All contributions by Tri Dao:
Copyright (c) 2024 Tri Dao.
All rights reserved.

All contributions by Arm:
Copyright (c) 2021, 2023-2024 Arm Limited and/or its affiliates

All contributions from Caffe:
Copyright(c) 2013, 2014, 2015, the respective contributors
All rights reserved.

All other contributions:
Copyright(c) 2015, 2016 the respective contributors
All rights reserved.

Caffe2 uses a copyright model similar to Caffe: each contributor holds
copyright over their contributions to Caffe2. The project versioning records
all such contribution and copyright details. If a contributor wants to further
mark their specific copyright on a particular contribution, they should
indicate their copyright solely in the commit message of the change when it is
committed.

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/*
 * Modified from c10::SizesAndStrides
 * - simplified implementation
 * - use std::array instead of raw array
 * - use copy_n, fill_n for memset and memcpy
 */

/*
 * view_pack is a class inspired (a lot) by PyTorch's `SizesAndStrides`.
 * Memory Layout:
 * - For tensors with dimension <= LEGRAD_VIEW_PACK_MAX_DIM (e.g., 5):
 *   [shape[0], ..., shape[4], stride[0], ..., stride[4]] - Stored inline
 * - For tensors with dimension > LEGRAD_VIEW_PACK_MAX_DIM:
 *   Out-of-line storage (dynamically allocated array) is used to store
 *   shape and stride data contiguously.
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "internal/array_view.h"
#include "macros/log.h"

using Int = int64_t;

namespace legrad::internal
{
static constexpr Int LEGRAD_VIEW_PACK_MAX_DIM = 5;

class view_pack
{
public:
  ~view_pack()
  {
    if (!is_inline()) {
      std::free(out_of_line_storage_);
    }
  }

  view_pack()
      : dim_(1)
  {
    inline_storage_.fill(0);
  }

  view_pack(size_t size)
      : dim_(size)
  {
    if (is_inline()) {
      inline_storage_.fill(0);
    } else {
      out_of_line_storage_ = allocate_new_storage(dim_);
      std::fill_n(out_of_line_storage_, dim_ * 2, 0);
    }
  }

  view_pack(const view_pack&);
  view_pack& operator=(const view_pack&);
  view_pack(view_pack&&) noexcept;
  view_pack& operator=(view_pack&&) noexcept;

  IntArrayView shape_view() const noexcept { return {shape_data(), dim()}; }
  IntArrayView stride_view() const noexcept { return {stride_data(), dim()}; }

  const Int* shape_data() const noexcept
  {
    return is_inline() ? &inline_storage_[0] : &out_of_line_storage_[0];
  }

  Int* shape_data() noexcept
  {
    return is_inline() ? &inline_storage_[0] : &out_of_line_storage_[0];
  }

  const Int* stride_data() const noexcept
  {
    return is_inline() ? &inline_storage_[LEGRAD_VIEW_PACK_MAX_DIM]
                       : &out_of_line_storage_[dim()];
  }

  Int* stride_data() noexcept
  {
    return is_inline() ? &inline_storage_[LEGRAD_VIEW_PACK_MAX_DIM]
                       : &out_of_line_storage_[dim()];
  }

  Int* shape_begin() { return shape_data(); }
  const Int* shape_begin() const { return shape_data(); }
  Int* shape_end() { return shape_data() + dim(); }
  const Int* shape_end() const { return shape_data() + dim(); }
  Int* stride_begin() { return stride_data(); }
  const Int* stride_begin() const { return stride_data(); }
  Int* stride_end() { return stride_data() + dim(); }
  const Int* stride_end() const { return stride_data() + dim(); }

  Int shape_at(size_t idx) const
  {
    LEGRAD_ASSERT(idx < dim_, "Index {} is out of range [0:{}) for shape", idx,
                  dim_);
    return unsafe_shape_at(idx);
  }

  Int stride_at(size_t idx) const
  {
    LEGRAD_ASSERT(idx < dim_, "Index {} is out of range [0:{}) for stride", idx,
                  dim_);
    return unsafe_stride_at(idx);
  }

  void set_shape(IntArrayView shape)
  {
    resize_storage(shape.size());
    std::copy(shape.begin(), shape.end(), shape_begin());
  }

  void set_stride(IntArrayView stride)
  {
    if (stride.size() != dim_) {
      LEGRAD_THROW_ERROR(std::invalid_argument,
                         "New stride is not match with current shape size", 0);
    }
    std::copy(stride.begin(), stride.end(), stride_begin());
  }

  void resize_storage(size_t new_dim);
  bool is_inline() const { return dim_ <= LEGRAD_VIEW_PACK_MAX_DIM; }
  size_t dim() const noexcept { return dim_; }

private:
  Int& unsafe_stride_at(size_t idx) { return stride_data()[idx]; }
  Int& unsafe_shape_at(size_t idx) { return shape_data()[idx]; }
  Int unsafe_stride_at(size_t idx) const { return stride_data()[idx]; }
  Int unsafe_shape_at(size_t idx) const { return shape_data()[idx]; }

  void resize_out_of_line_storage(size_t new_dim, size_t old_dim);
  void move_out_to_inline_storage(size_t new_dim, size_t old_dim);
  void move_inline_to_out_storage(size_t new_dim, size_t old_dim);

  Int* allocate_new_storage(size_t n);
  void reallocate_out_of_line_storage(size_t n);
  Int storage_bytes(size_t n) noexcept { return n * 2 * sizeof(Int); }

  void copy_inline_storage(const view_pack& other)
  {
    std::copy(other.inline_storage_.begin(), other.inline_storage_.end(),
              inline_storage_.data());
  }

  void copy_outline_storage(const view_pack& other)
  {
    std::copy(other.out_of_line_storage_,
              other.out_of_line_storage_ + other.dim() * 2,
              out_of_line_storage_);
  }

private:
  size_t dim_;
  union
  {
    std::array<Int, LEGRAD_VIEW_PACK_MAX_DIM * 2> inline_storage_;
    Int* out_of_line_storage_;
  };
};
}  // namespace legrad::internal