#include "ops.h"
#include <stdexcept>
std::unique_ptr<Tensor> conv2d(
    const Tensor& input,
    const Tensor& weights,
    int stride,
    int padding
) {
int in_channels  = input.shape[0];
int in_height    = input.shape[1];
int in_width     = input.shape[2];

int out_channels = weights.shape[0];
int weight_in_ch = weights.shape[1];
int kernel_h     = weights.shape[2];
int kernel_w     = weights.shape[3];











}