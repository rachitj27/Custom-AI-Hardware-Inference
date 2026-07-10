#ifndef OPS_H
#define OPS_H

#include "tensor.h"
#include <memory>

// Perform 2D convolution.
// input:  shape (in_channels, in_height, in_width)  [assume batch = 1]
// weights: shape (out_channels, in_channels, kernel_h, kernel_w)
// stride, padding: same as PyTorch
// Returns: a new Tensor with shape (out_channels, out_height, out_width)
std::unique_ptr<Tensor> conv2d(
    const Tensor& input,
    const Tensor& weights,
    int stride,
    int padding
);

#endif