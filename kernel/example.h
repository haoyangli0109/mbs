#pragma once

#include <cstdint>

// Pure C++ element-wise add kernel operating on raw data pointers.
// No torch / ATen dependency here -- this is just arithmetic over buffers.
//
//   out[i] = a[i] + b[i]   for i in [0, n)
//
// Defined for float and double via explicit instantiation in tensor_add.cpp.
template <typename T>
void tensor_add(const T* a, const T* b, T* out, int64_t n);

void read_pattern(const uint8_t* x, size_t x_stride, uint8_t* y);
void mbs_factor_experiment();
