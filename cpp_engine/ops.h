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
    int padding,
    float input_scale,
    float weight_scale,
    float output_scale,
    int output_zero_point
);

void silu_inplace(
    Tensor& tensor,
    float input_scale,
    int input_zero_point,
    float output_scale,
    int output_zero_point
);
std::unique_ptr<Tensor> upsample2x(const Tensor& input);
std::unique_ptr<Tensor> concat(const std::vector<const Tensor*>& tensors);
std::unique_ptr<Tensor> maxpool2d(const Tensor& input, int kernel_size);

#endif