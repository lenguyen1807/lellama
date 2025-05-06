#pragma once

#include "internal/enum_impl.h"

namespace legrad::core
{
LEGRAD_ENUM(TypeInfo,
            uint8_t,  // type of Enum
            Bool,  // Begin of Enum
            Float32,  // End of Enum
            Bool,
            UInt8,
            UInt16,
            UInt32,
            Int8,
            Int16,
            Int32,
            Float16,
            // BFloat16, // will support in the future
            Float32,
            COUNT)
LEGRAD_ENUM(TypeKind, uint8_t, Bool, Float, Bool, Uint, Int, Float, COUNT)

// clang-format off
#define CALL_DISPATCH_TYPE_INFO(TYPE, ...)          \
    [&] {                                           \
        switch (TYPE) {                             \
            case TypeInfo::Float16: {               \
                using scalar_t = half_float;        \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Float32: {               \
                using scalar_t = float;             \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Int8: {                  \
                using scalar_t = int8_t;            \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Int16: {                 \
                using scalar_t = int16_t;           \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Int32: {                 \
                using scalar_t = int32_t;           \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::UInt8: {                 \
                using scalar_t = uint8_t;           \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::UInt16: {                \
                using scalar_t = uint16_t;          \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::UInt32: {                \
                using scalar_t = uint32_t;          \
                return __VA_ARGS__();               \
            }                                       \
            case TypeInfo::Bool: {                  \
                using scalar_t = bool;              \
                return __VA_ARGS__();               \
            }                                       \
            default:                                \
                LEGRAD_THROW_ERROR(std::runtime_error,\
                "Unsupported TypeInfo", 0);         \
        }                                           \
    }()
}