#ifndef KERNEL_EXAMPLE_H_
#define KERNEL_EXAMPLE_H_

// Disabling all the warnings from Tensorflow
#pragma warning(push, 0)
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#pragma warning(pop)

template <typename Device, typename T>
struct InteractiveFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#endif KERNEL_EXAMPLE_H_