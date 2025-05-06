#pragma once

#include <cstddef>
#include <fstream>
#include <istream>
#include <map>
#include <string>
#include <type_traits>

#include "gguf_def.h"
#include "macros/expr.h"
#include "macros/log.h"

namespace legrad::gguf
{
template <typename T>
gguf_type type_to_gguf_type()
{
  if constexpr (std::is_same_v<T, float>)
    return GGUF_TYPE_FLOAT32;

  if constexpr (std::is_same_v<T, double>)
    return GGUF_TYPE_FLOAT64;

  if constexpr (std::is_same_v<T, int8_t>)
    return GGUF_TYPE_INT8;

  if constexpr (std::is_same_v<T, int16_t>)
    return GGUF_TYPE_INT16;

  if constexpr (std::is_same_v<T, int32_t>)
    return GGUF_TYPE_INT32;

  if constexpr (std::is_same_v<T, int64_t>)
    return GGUF_TYPE_INT64;

  if constexpr (std::is_same_v<T, uint8_t>)
    return GGUF_TYPE_UINT8;

  if constexpr (std::is_same_v<T, uint16_t>)
    return GGUF_TYPE_UINT16;

  if constexpr (std::is_same_v<T, uint32_t>)
    return GGUF_TYPE_UINT32;

  if constexpr (std::is_same_v<T, uint64_t>)
    return GGUF_TYPE_UINT64;

  if constexpr (std::is_same_v<T, bool>)
    return GGUF_TYPE_BOOL;

  if constexpr (std::is_same_v<T, std::string>)
    return GGUF_TYPE_STRING;
}

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
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

static const std::map<gguf_type, const char*> GGUF_TYPE_NAME = {
    {GGUF_TYPE_UINT8, "u8"},    {GGUF_TYPE_INT8, "i8"},
    {GGUF_TYPE_UINT16, "u16"},  {GGUF_TYPE_INT16, "i16"},
    {GGUF_TYPE_UINT32, "u32"},  {GGUF_TYPE_INT32, "i32"},
    {GGUF_TYPE_FLOAT32, "f32"}, {GGUF_TYPE_BOOL, "bool"},
    {GGUF_TYPE_STRING, "str"},  {GGUF_TYPE_ARRAY, "arr"},
    {GGUF_TYPE_UINT64, "u64"},  {GGUF_TYPE_INT64, "i64"},
    {GGUF_TYPE_FLOAT64, "f64"},
};
static_assert(GGUF_TYPE_COUNT == 13, "GGUF_TYPE_COUNT != 13");

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
      , type(type_to_gguf_type<T>())
  {
    LEGRAD_DEFAULT_ASSERT(!key.empty());
    data.resize(sizeof(T));
    memcpy(data.data(), &value, sizeof(T));
  }

  template <typename T>
  gguf_kv(const std::string& key, const std::vector<T>& value)
      : key(key)
      , is_array(true)
      , type(type_to_gguf_type<T>())
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
    LEGRAD_DEFAULT_ASSERT(type_to_gguf_type<T>() == type);

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

struct gguf_tensor_info
{
  struct ggml_tensor t;  // for holding the equivalent info
  uint64_t
      offset;  // offset from start of `data`, must be a multiple of `ALIGNMENT`
};

struct gguf_context
{
  uint32_t version = GGUF_VERSION;

  std::vector<struct gguf_kv> kv;
  std::vector<struct gguf_tensor_info> info;

  size_t alignment = GGUF_DEFAULT_ALIGNMENT;
  size_t offset = 0;  // offset of `data` from beginning of file
  size_t size = 0;  // size of `data` in bytes

  void* data = nullptr;
};

struct gguf_reader
{
  FILE* file;

  gguf_reader(FILE* file)
      : file(file)
  {
  }

  template <typename T>
  bool read(T& dst) const
  {
    return fread(&dst, 1, sizeof(dst), file) == sizeof(dst);
  }

  template <typename T>
  bool read(std::vector<T>& dst, const size_t n) const
  {
    dst.resize(n);
    for (size_t i = 0; i < dst.size(); ++i) {
      if constexpr (std::is_same<T, bool>::value) {
        bool tmp;
        if (!read(tmp)) {
          return false;
        }
        dst[i] = tmp;
      } else {
        if (!read(dst[i])) {
          return false;
        }
      }
    }
    return true;
  }

  bool read(bool& dst) const
  {
    int8_t tmp = -1;
    if (!read(tmp)) {
      return false;
    }
    dst = tmp != 0;
    return true;
  }

  bool read(enum ggml_type& dst) const
  {
    int32_t tmp = -1;
    if (!read(tmp)) {
      return false;
    }
    dst = ggml_type(tmp);
    return true;
  }

  bool read(enum gguf_type& dst) const
  {
    int32_t tmp = -1;
    if (!read(tmp)) {
      return false;
    }
    dst = gguf_type(tmp);
    return true;
  }

  bool read(std::string& dst) const
  {
    uint64_t size = -1;
    if (!read(size)) {
      return false;
    }
    dst.resize(size);
    return fread(dst.data(), 1, dst.length(), file) == dst.length();
  }

  bool read(void* dst, const size_t size) const
  {
    return fread(dst, 1, size, file) == size;
  }
};

template <typename T>
bool gguf_read_emplace_helper(const struct gguf_reader& gr,
                              std::vector<struct gguf_kv>& kv,
                              const std::string& key,
                              const bool is_array,
                              const size_t n)
{
  if (is_array) {
    std::vector<T> value;
    try {
      if (!gr.read(value, n)) {
        return false;
      }
    } catch (std::length_error&) {
      LEGRAD_LOG_ERR("Encounted length_error while reading value for key {}",
                     key);
      return false;
    } catch (std::bad_alloc&) {
      LEGRAD_LOG_ERR("Encounted bad_alloc while reading value for key {}", key);
      return false;
    }
    kv.emplace_back(key, value);
  } else {
    T value;
    if (!gr.read(value)) {
      return false;
    }
    kv.emplace_back(key, value);
  }
  return true;
}
}  // namespace legrad::gguf