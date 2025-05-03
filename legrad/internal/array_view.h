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

// legrad: modified from c10::ArrayRef
// - simplified implementation

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "macros/log.h"

namespace legrad::internal
{
// this is a tag to indicate slice count (from elem i and to i + n for count =
// n)
struct SliceRange
{
};

template <typename T>
class array_view
{
  // --- Member Types ---
  using value_type = T;
  using pointer = const T*;
  using const_pointer = const T*;
  using reference = const T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_iterator = const T*;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

public:
  // --- Constructors ---
  /// Construct an empty array_view.
  constexpr array_view() noexcept
      : ptr_(nullptr)
      , len_(0)
  {
  }

  /// Construct an array_view from a single element.
  /* implicit */ constexpr array_view(
      const T& elem) noexcept  // Assuming taking address is noexcept
      : ptr_(&elem)
      , len_(1)
  {
  }

  /// Construct an array_view from a pointer and length.
  constexpr array_view(const T* ptr, std::size_t len) noexcept
      : ptr_(ptr)
      , len_(len)
  {
    debug_check_nullptr();
  }

  /// Construct an array_view from a range (start and end pointers).
  constexpr array_view(const T* start, const T* end) noexcept
      : ptr_(start)
      , len_(static_cast<size_type>(end - start))  // Calculate length safely
  {
    LEGRAD_ASSERT(start <= end, "Start pointer cannot be after end pointer", 0);
    debug_check_nullptr();
  }

  /// Construct an array_view from a C-style array.
  template <size_t N>
  // NOLINTNEXTLINE(*c-arrays*) - Necessary for this constructor
  /* implicit */ constexpr array_view(const T (&Arr)[N]) noexcept
      : ptr_(Arr)
      , len_(N)
  {
    // No debugCheck needed if N > 0, C-arrays can't be null.
    // If N == 0, ptr_ might be anything but len_ is 0, so invariant holds.
  }

  /// Construct an array_view from any contiguous container defining data() and
  /// size() (e.g., std::vector, std::array, std::string).
  template <
      typename Container,
      typename = std::enable_if_t<
          std::is_same_v<
              std::remove_const_t<decltype(std::declval<Container>().data())>,
              T*>
          && std::is_convertible_v<decltype(std::declval<Container>().size()),
                                   std::size_t>
          // Add more checks if needed (e.g., contiguity tags in C++20)
          >>
  /* implicit */ constexpr array_view(const Container& container) noexcept
      : ptr_(container.data())
      , len_(static_cast<size_type>(container.size()))  // Cast size() result
  {
    // Implicitly handles std::vector, std::array, etc.
    static_assert(
        !std::is_same_v<Container,
                        std::vector<bool>>,  // Disallow vector<bool> proxy
        "Cannot construct array_view from std::vector<bool>");
    debug_check_nullptr();
  }

  /// Construct an array_view from a std::initializer_list.
  /* implicit */ constexpr array_view(
      const std::initializer_list<T>& list) noexcept
      : ptr_(std::begin(list) == std::end(list) ? nullptr : std::begin(list))
      , len_(list.size())
  {
    // Invariant holds: if list is empty, ptr is nullptr, len is 0.
    // If list non-empty, ptr is valid, len > 0.
  }

  // --- Deleted Assignment ---
  // Prevent accidental rebinding of the view via assignment,
  // allowing only construction and copy/move construction/assignment.
  // "view = {}" or "view = other_view" remain valid.
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, array_view<T>&> operator=(
      // NOLINTNEXTLINE(cppcoreguidelines-missing-std-forward)
      U&& Temporary) = delete;
  template <typename U>
  std::enable_if_t<std::is_same_v<U, T>, array_view<T>&> operator=(
      std::initializer_list<U>) = delete;

  // --- Observers ---

  constexpr const T& operator[](size_t idx) const noexcept
  {
    // Standard practice for operator[] is no bounds check for performance.
    // Use .at() for checked access.
    return ptr_[idx];
  }

  const T& at(size_t idx) const
  {
    LEGRAD_ASSERT(idx < len_,
                  "Index {} is out of bounds for array_view of size {}", idx,
                  len_);
    return ptr_[idx];
  }

  constexpr const T* data() const noexcept { return ptr_; }
  constexpr size_t size() const noexcept { return len_; }
  constexpr bool empty() const noexcept { return len_ == 0; }

  // --- Iterators ---
  constexpr iterator begin() const noexcept { return ptr_; }
  constexpr iterator end() const noexcept { return ptr_ + len_; }
  constexpr const_iterator cbegin() const noexcept { return ptr_; }
  constexpr const_iterator cend() const noexcept { return ptr_ + len_; }

  constexpr reverse_iterator rbegin() const noexcept
  {
    return std::make_reverse_iterator(end());
  }
  constexpr reverse_iterator rend() const noexcept
  {
    return std::make_reverse_iterator(begin());
  }

  // --- Element Access ---
  constexpr const T& front() const noexcept
  {
    LEGRAD_ASSERT(!empty(), "Attempt to access front() of empty array_view", 0);
    return ptr_[0];
  }

  constexpr const T& back() const noexcept
  {
    LEGRAD_ASSERT(!empty(), "Attempt to access back() of empty array_view", 0);
    return ptr_[len_ - 1];
  }

  // Returns a view of the subarray [start, end).
  constexpr array_view<T> slice(size_t start, size_t end_pos) const noexcept
  {
    LEGRAD_ASSERT(start <= end_pos,
                  "Slice start index {} cannot be greater than end index {}",
                  start, end_pos);
    LEGRAD_ASSERT(end_pos <= size(),
                  "Slice end index {} cannot be greater than view size {}",
                  end_pos, size());
    return array_view<T>(data() + start, end_pos - start);
  }

  /// Returns a view of the subarray starting at `start_index` with `count`
  /// elements.
  constexpr array_view<T> slice(size_t start_index,
                                size_t count,
                                SliceRange) const noexcept
  {
    LEGRAD_ASSERT(start_index + count <= size(),
                  "Slice start index {} + count {} exceeds view size {}",
                  start_index, count, size());
    return array_view<T>(data() + start_index, count);
  }

  /// Returns a view of the subarray starting at `start_index` until the end.
  constexpr array_view<T> slice(size_t start_index) const noexcept
  {
    LEGRAD_ASSERT(start_index <= size(),
                  "Slice start index {} cannot be greater than view size {}",
                  start_index, size());
    return array_view<T>(data() + start_index, size() - start_index);
  }

  // Converts the view to a std::vector (expensive - copies data).
  std::vector<T> to_vec() const { return std::vector<T>(ptr_, ptr_ + len_); }

  // --- Comparison ---
  bool equals(array_view other) const noexcept
  {
    if (len_ != other.len_) {
      return false;
    }
    return std::equal(begin(), end(), other.begin());
  }

  bool equals(const std::initializer_list<T>& other) const noexcept
  {
    return equals(array_view<T>(other));
  }

  // --- Output ---
  friend std::ostream& operator<<(std::ostream& os, array_view view)
  {
    os << numerical_view_2str(view);
    return os;
  }

  // --- Static Helpers ---
  static std::string numerical_view_2str(internal::array_view<T> view)
  {
    LEGRAD_DEFAULT_ASSERT(std::is_arithmetic_v<T>);  // Keep if intended

    if (view.empty()) {
      return "[]";  // More conventional output for empty
    }

    std::string result = "[";

    bool first = true;
    for (const auto& item : view) {
      if (!first) {
        result += ", ";
      }
      result += std::to_string(item);
      first = false;
    }
    result += "]";

    return result;
  }

private:
  const T* ptr_;
  size_t len_;

  // Helper to check the invariant: ptr_ should not be null if len_ > 0.
  constexpr void debug_check_nullptr() const noexcept
  {
    LEGRAD_ASSERT(
        ptr_ != nullptr || len_ == 0,
        "Invariant violation: array_view has nullptr data but non-zero length",
        0);
  }
};

template <typename T>
bool operator==(
    internal::array_view<T> lhs,
    internal::array_view<T> rhs) noexcept  // Assuming T::operator== is noexcept
{
  return lhs.equals(rhs);
}

template <typename T>
bool operator!=(
    internal::array_view<T> lhs,
    internal::array_view<T> rhs) noexcept  // Assuming T::operator!= is noexcept
{
  return !(lhs == rhs);
}

// Comparisons with std::vector (optional, but convenient)
template <typename T>
bool operator==(const std::vector<T>& lhs, internal::array_view<T> rhs) noexcept
{
  return internal::array_view<T>(lhs).equals(rhs);
}

template <typename T>
bool operator==(internal::array_view<T> lhs, const std::vector<T>& rhs) noexcept
{
  return lhs.equals(internal::array_view<T>(rhs));
}

template <typename T>
bool operator!=(const std::vector<T>& lhs, internal::array_view<T> rhs) noexcept
{
  return !(lhs == rhs);
}

template <typename T>
bool operator!=(internal::array_view<T> lhs, const std::vector<T>& rhs) noexcept
{
  return !(lhs == rhs);
}
}  // namespace legrad::internal

namespace legrad
{
// Public aliases
template <typename T>
using ArrayView = internal::array_view<T>;

using IntArrayView = ArrayView<int64_t>;
using Int2DArrayView = ArrayView<IntArrayView>;
}  // namespace legrad