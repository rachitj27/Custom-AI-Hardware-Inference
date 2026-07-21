#ifndef OPS_H
#define OPS_H
#include "model.h"

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
std::unique_ptr<Tensor> bottleneck(
    const Tensor& input,
    const Layer& conv1, float input_scale, int input_zp, float mid_scale, int mid_zp,
    const Layer& conv2, float out_scale, int out_zp,
    bool shortcut
);

std::unique_ptr<Tensor> c2f_block(
    const Tensor& input,
    const std::vector<const Layer*>& convs,
    const std::vector<float>& scales,
    const std::vector<int>& zero_points,
    int n_bottlenecks,
    bool shortcut
);

std::unique_ptr<Tensor> sppf_block(
    const Tensor& input,
    const Layer& cv1, float input_scale, int input_zp, float mid_scale, int mid_zp,
    const Layer& cv2, float out_scale, int out_zp,
    int kernel_size
);

#endif