#pragma once

#include <cstddef>
#include <cstdint>

namespace legrad::def
{
using half = uint16_t;
using half2 = uint32_t;

// 16 bytes is sufficent for ARM
constexpr size_t MEMORY_ALIGNMENT_SIZE = 16;

constexpr size_t PAGE_SIZE =
    4000;  // A page in Metal is 4Kb (this is assumption from NVIDIA CUDA)

constexpr size_t MAX_BUCKET_SIZE = 6;
constexpr size_t BUCKET_SIZES[MAX_BUCKET_SIZE] = {64,   128,  256,
                                                  1024, 2048, 4096};
}  // namespace legrad::def