/**
* This file contains functionality related to "GGUF" files, the binary file
format used by ggml.
* GGUF files have the following structure:
*
* 1. File magic "GGUF" (4 bytes).
* 2. File version (uint32_t).
* 3. Number of ggml tensors in file (int64_t).
* 4. Number of key-value-pairs in file (int64_t).
* 5. For each KV pair:
*   1. The key (string).
*   2. The value type (gguf_type).
*   3a. If the value type is GGUF_TYPE_ARRAY:
*     1. The type of the array (gguf_type).
*     2. The number of elements in the array (uint64_t).
*     3. The binary representation of each element in the array.
*   3b. Otherwise:
*     1. The binary representation of the value.
* 6. For each ggml tensor:
*   1. The tensor name (string).
*   2. The number of dimensions of the tensor (uint32_t).
*   3. For each dimension:
*     1. The size of the tensor in the dimension (int64_t).
*   4. The tensor data type (ggml_type).
*   5. The tensor data offset in the tensor data binary blob (uint64_t).
* 7. The tensor data binary blob (optional, aligned).
*
* Strings are serialized as the string length (uint64_t) followed by the C
string without the null terminator.
* All enums are stored as int32_t.
* All bool values are stored as int8_t.
* If the special key "general.alignment" (uint32_t) is defined it is used for
alignment,
*   otherwise GGUF_DEFAULT_ALIGNMENT is used.
*
* Module maintainer: Johannes Gäßler (@JohannesGaessler, johannesg@5d6.de)
*/

/**
 * Modified from GGUF reader from llama.cpp
 */

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "macros/log.h"

namespace legrad::gguf
{
constexpr uint32_t GGUF_MAGIC =
    1179993927;  // little edian, hex number: 46 55 47 47 = F U G G => GGUF
constexpr uint32_t GGUF_FILE_VERSION = 3;
constexpr uint32_t GGUF_MAX_DIMENSION = 5;

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

template <typename T>
struct type_to_gguf_type;

template <>
struct type_to_gguf_type<uint8_t>
{
  static constexpr enum gguf_type value = GGUF_TYPE_UINT8;
};

template <>
struct type_to_gguf_type<int8_t>
{
  static constexpr enum gguf_type value = GGUF_TYPE_INT8;
};

template <>
struct type_to_gguf_type<uint16_t>
{
  static constexpr enum gguf_type value = GGUF_TYPE_UINT16;
};

template <>
struct type_to_gguf_type<int16_t>
{
  static constexpr enum gguf_type value = GGUF_TYPE_INT16;
};

template <>
struct type_to_gguf_type<uint32_t>
{
  static constexpr enum gguf_type value = GGUF_TYPE_UINT32;
};

template <>
struct type_to_gguf_type<int32_t>
{
  static constexpr enum gguf_type value = GGUF_TYPE_INT32;
};

template <>
struct type_to_gguf_type<float>
{
  static constexpr enum gguf_type value = GGUF_TYPE_FLOAT32;
};

template <>
struct type_to_gguf_type<bool>
{
  static constexpr enum gguf_type value = GGUF_TYPE_BOOL;
};

template <>
struct type_to_gguf_type<std::string>
{
  static constexpr enum gguf_type value = GGUF_TYPE_STRING;
};

template <>
struct type_to_gguf_type<uint64_t>
{
  static constexpr enum gguf_type value = GGUF_TYPE_UINT64;
};

template <>
struct type_to_gguf_type<int64_t>
{
  static constexpr enum gguf_type value = GGUF_TYPE_INT64;
};

template <>
struct type_to_gguf_type<double>
{
  static constexpr enum gguf_type value = GGUF_TYPE_FLOAT64;
};

static const std::map<gguf_type, size_t> GGUF_TYPE_SIZE = {
    {GGUF_TYPE_UINT8, sizeof(uint8_t)},
    {GGUF_TYPE_INT8, sizeof(int8_t)},
    {GGUF_TYPE_UINT16, sizeof(uint16_t)},
    {GGUF_TYPE_INT16, sizeof(int16_t)},
    {GGUF_TYPE_UINT32, sizeof(uint32_t)},
    {GGUF_TYPE_INT32, sizeof(int32_t)},
    {GGUF_TYPE_FLOAT32, sizeof(float)},
    {GGUF_TYPE_BOOL, sizeof(int8_t)},
    {GGUF_TYPE_STRING, 0},  // undefined
    {GGUF_TYPE_ARRAY, 0},  // undefined
    {GGUF_TYPE_UINT64, sizeof(uint64_t)},
    {GGUF_TYPE_INT64, sizeof(int64_t)},
    {GGUF_TYPE_FLOAT64, sizeof(double)},
};

static const std::map<gguf_type, const char*> GGUF_TYPE_NAME = {
    {GGUF_TYPE_UINT8, "u8"},    {GGUF_TYPE_INT8, "i8"},
    {GGUF_TYPE_UINT16, "u16"},  {GGUF_TYPE_INT16, "i16"},
    {GGUF_TYPE_UINT32, "u32"},  {GGUF_TYPE_INT32, "i32"},
    {GGUF_TYPE_FLOAT32, "f32"}, {GGUF_TYPE_BOOL, "bool"},
    {GGUF_TYPE_STRING, "str"},  {GGUF_TYPE_ARRAY, "arr"},
    {GGUF_TYPE_UINT64, "u64"},  {GGUF_TYPE_INT64, "i64"},
    {GGUF_TYPE_FLOAT64, "f64"},
};

LEGRAD_INLINE size_t gguf_type_size(enum gguf_type type)
{
  auto it = GGUF_TYPE_SIZE.find(type);
  return it == GGUF_TYPE_SIZE.end() ? 0 : it->second;
}

struct gguf_kv
{
  std::string key;
  bool is_array;
  enum gguf_type type;

  std::vector<int8_t> data;
  std::vector<std::string> data_string;

  template <typename T>
  gguf_kv(const std::string& key, const T value)
      : key(key)
      , is_array(false)
      , type(type_to_gguf_type<T>::value)
  {
    LEGRAD_DEFAULT_ASSERT(!key.empty());
    data.resize(sizeof(T));
    memcpy(data.data(), &value, sizeof(T));
  }

  template <typename T>
  gguf_kv(const std::string& key, const std::vector<T>& value)
      : key(key)
      , is_array(true)
      , type(type_to_gguf_type<T>::value)
  {
    LEGRAD_DEFAULT_ASSERT(!key.empty());
    data.resize(value.size() * sizeof(T));
    for (size_t i = 0; i < value.size(); ++i) {
      const T tmp = value[i];
      memcpy(data.data() + i * sizeof(T), &tmp, sizeof(T));
    }
  }

  gguf_kv(const std::string& key, const std::string& value)
      : key(key)
      , is_array(false)
      , type(GGUF_TYPE_STRING)
  {
    LEGRAD_DEFAULT_ASSERT(!key.empty());
    data_string.push_back(value);
  }

  gguf_kv(const std::string& key, const std::vector<std::string>& value)
      : key(key)
      , is_array(true)
      , type(GGUF_TYPE_STRING)
  {
    LEGRAD_DEFAULT_ASSERT(!key.empty());
    data_string = value;
  }

  const std::string& get_key() const { return key; }

  const enum gguf_type& get_type() const { return type; }

  size_t get_ne() const
  {
    if (type == GGUF_TYPE_STRING) {
      const size_t ne = data_string.size();
      LEGRAD_DEFAULT_ASSERT(is_array || ne == 1);
      return ne;
    }
    const size_t type_size = gguf_type_size(type);
    LEGRAD_DEFAULT_ASSERT(data.size() % type_size == 0);
    const size_t ne = data.size() / type_size;
    LEGRAD_DEFAULT_ASSERT(is_array || ne == 1);
    return ne;
  }

  template <typename T>
  const T& get_val(const size_t i = 0) const
  {
    LEGRAD_DEFAULT_ASSERT(type_to_gguf_type<T>::value == type);
    if constexpr (std::is_same<T, std::string>::value) {
      LEGRAD_DEFAULT_ASSERT(data_string.size() >= i + 1);
      return data_string[i];
    }
    const size_t type_size = gguf_type_size(type);
    LEGRAD_DEFAULT_ASSERT(data.size() % type_size == 0);
    LEGRAD_DEFAULT_ASSERT(data.size() >= (i + 1) * type_size);
    return reinterpret_cast<const T*>(data.data())[i];
  }

  void cast(const enum gguf_type new_type)
  {
    const size_t new_type_size = gguf_type_size(new_type);
    LEGRAD_DEFAULT_ASSERT(data.size() % new_type_size == 0);
    type = new_type;
  }
};
}  // namespace legrad::gguf
