#include <metal_stdlib>

using namespace metal;

/* --- Add Kernel --- */
// https://stackoverflow.com/questions/56687496/how-to-make-templated-compute-kernels-in-metal
template<typename T>
kernel void add(
        device const T * src0,
        device const T * src1,
        device T * dst,
        uint id [[thread_position_in_grid]]) {
    dst[id] = src0[id] + src1[id];
}

// F64
template [[ host_name("add_double") ]] 
kernel void add(device const double*, device const double*, device double*, uint);

// F32
template [[ host_name("add_float") ]] 
kernel void add(device const float*, device const float*, device float*, uint);

// F16
template [[ host_name("add_host") ]]
kernel void add(device const half*, device const half*, device half*, uint);