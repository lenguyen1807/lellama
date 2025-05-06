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

// F32
template [[ host_name("Add_Float32") ]] 
kernel void add(device const float*, device const float*, device float*, uint);

// F16
template [[ host_name("Add_Float16") ]]
kernel void add(device const half*, device const half*, device half*, uint);

// BF16
template [[ host_name("Add_BFloat16") ]]
kernel void add(device const bfloat*, device const bfloat*, device bfloat*, uint);

/* --- Sub Kernel --- */
// https://stackoverflow.com/questions/56687496/how-to-make-templated-compute-kernels-in-metal
template<typename T>
kernel void sub(
        device const T * src0,
        device const T * src1,
        device T * dst,
        uint id [[thread_position_in_grid]]) {
    dst[id] = src0[id] - src1[id];
}

// F32
template [[ host_name("Sub_Float32") ]] 
kernel void sub(device const float*, device const float*, device float*, uint);

// F16
template [[ host_name("Sub_Float16") ]]
kernel void sub(device const half*, device const half*, device half*, uint);

// BF16
template [[ host_name("Sub_BFloat16") ]]
kernel void sub(device const bfloat*, device const bfloat*, device bfloat*, uint);