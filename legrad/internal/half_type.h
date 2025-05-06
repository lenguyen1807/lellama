#pragma once

#include <cstdint>

#include "fp16/fpt16.h"

namespace legrad::internal
{
struct half
{
  uint16_t raw_bits;

  // clang-format off
  half(float value) 
  { 
    raw_bits = fp16_ieee_from_fp32_value(value); 
  }

  uint32_t to_float32_bits() 
  { 
    return fp16_ieee_to_fp32_bits(raw_bits); 
  }

  float to_float32() 
  { 
    return fp16_ieee_to_fp32_value(raw_bits); 
  }
};
}

using half_float = legrad::internal::half;