// This file contains functionality related to "GGUF" files, the binary file
// format used by ggml. GGUF files have the following structure:
//
// 1. File magic "GGUF" (4 bytes).
// 2. File version (uint32_t).
// 3. Number of ggml tensors in file (int64_t).
// 4. Number of key-value-pairs in file (int64_t).
// 5. For each KV pair:
//   1. The key (string).
//   2. The value type (gguf_type).
//   3a. If the value type is GGUF_TYPE_ARRAY:
//     1. The type of the array (gguf_type).
//     2. The number of elements in the array (uint64_t).
//     3. The binary representation of each element in the array.
//   3b. Otherwise:
//     1. The binary representation of the value.
// 6. For each ggml tensor:
//   1. The tensor name (string).
//   2. The number of dimensions of the tensor (uint32_t).
//   3. For each dimension:
//     1. The size of the tensor in the dimension (int64_t).
//   4. The tensor data type (ggml_type).
//   5. The tensor data offset in the tensor data binary blob (uint64_t).
// 7. The tensor data binary blob (optional, aligned).
//
// Strings are serialized as the string length (uint64_t) followed by the C
// string without the null terminator. All enums are stored as int32_t. All bool
// values are stored as int8_t. If the special key "general.alignment"
// (uint32_t) is defined it is used for alignment,
//   otherwise GGUF_DEFAULT_ALIGNMENT is used.
//
// Module maintainer: Johannes Gäßler (@JohannesGaessler, johannesg@5d6.de)

// Modification by legrad (github.com/lenguyen1807/legrad)

#pragma once

#include <cstddef>
#include <cstdint>

namespace legrad::gguf
{
#define GGUF_MAGIC "GGUF"
#define GGUF_VERSION 3
#define GGUF_KEY_GENERAL_ALIGNMENT "general.alignment"
#define GGUF_DEFAULT_ALIGNMENT 32
#define GGML_MAX_DIMS 4
#define GGML_MAX_SRC 10
#define GGML_MAX_NAME 64

// types that can be stored as GGUF KV data
enum gguf_type
{
  GGUF_TYPE_UINT8 = 0,
  GGUF_TYPE_INT8 = 1,
  GGUF_TYPE_UINT16 = 2,
  GGUF_TYPE_INT16 = 3,
  GGUF_TYPE_UINT32 = 4,
  GGUF_TYPE_INT32 = 5,
  GGUF_TYPE_FLOAT32 = 6,
  GGUF_TYPE_BOOL = 7,
  GGUF_TYPE_STRING = 8,
  GGUF_TYPE_ARRAY = 9,
  GGUF_TYPE_UINT64 = 10,
  GGUF_TYPE_INT64 = 11,
  GGUF_TYPE_FLOAT64 = 12,
  GGUF_TYPE_COUNT,  // marks the end of the enum
};

// NOTE: always add types at the end of the enum to keep backward compatibility
enum ggml_type
{
  GGML_TYPE_F32 = 0,
  GGML_TYPE_F16 = 1,
  GGML_TYPE_Q4_0 = 2,
  GGML_TYPE_Q4_1 = 3,
  // GGML_TYPE_Q4_2 = 4, support has been removed
  // GGML_TYPE_Q4_3 = 5, support has been removed
  GGML_TYPE_Q5_0 = 6,
  GGML_TYPE_Q5_1 = 7,
  GGML_TYPE_Q8_0 = 8,
  GGML_TYPE_Q8_1 = 9,
  GGML_TYPE_Q2_K = 10,
  GGML_TYPE_Q3_K = 11,
  GGML_TYPE_Q4_K = 12,
  GGML_TYPE_Q5_K = 13,
  GGML_TYPE_Q6_K = 14,
  GGML_TYPE_Q8_K = 15,
  GGML_TYPE_IQ2_XXS = 16,
  GGML_TYPE_IQ2_XS = 17,
  GGML_TYPE_IQ3_XXS = 18,
  GGML_TYPE_IQ1_S = 19,
  GGML_TYPE_IQ4_NL = 20,
  GGML_TYPE_IQ3_S = 21,
  GGML_TYPE_IQ2_S = 22,
  GGML_TYPE_IQ4_XS = 23,
  GGML_TYPE_I8 = 24,
  GGML_TYPE_I16 = 25,
  GGML_TYPE_I32 = 26,
  GGML_TYPE_I64 = 27,
  GGML_TYPE_F64 = 28,
  GGML_TYPE_IQ1_M = 29,
  GGML_TYPE_BF16 = 30,
  // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
  // GGML_TYPE_Q4_0_4_8 = 32,
  // GGML_TYPE_Q4_0_8_8 = 33,
  GGML_TYPE_TQ1_0 = 34,
  GGML_TYPE_TQ2_0 = 35,
  // GGML_TYPE_IQ4_NL_4_4 = 36,
  // GGML_TYPE_IQ4_NL_4_8 = 37,
  // GGML_TYPE_IQ4_NL_8_8 = 38,
  GGML_TYPE_COUNT = 39,
};

// simplified version of ggml_tensor
struct ggml_tensor
{
  enum ggml_type type;

  int64_t ne[GGML_MAX_DIMS];  // number of elements

  size_t nb[GGML_MAX_DIMS];  // stride in bytes:
  // nb[0] = ggml_type_size(type)
  // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type))
  // + padding nb[i] = nb[i-1] * ne[i-1]

  struct ggml_tensor* src[GGML_MAX_SRC];

  // source tensor and offset for views
  struct ggml_tensor* view_src;
  size_t view_offs;

  // read data
  void* data;
  void* extra;  // extra things e.g. for ggml-cuda.cu
  char padding[8];

  char name[GGML_MAX_NAME];
};
}  // namespace legrad::gguf