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

    if (in_channels != weight_in_ch) {
        throw std::runtime_error("Input channels do not match weight in_channels");
    }

    int out_height = (in_height + 2 * padding - kernel_h) / stride + 1;
    int out_width  = (in_width  + 2 * padding - kernel_w) / stride + 1;

    auto output = std::make_unique<Tensor>(std::vector<int>{out_channels, out_height, out_width});

    for (int oc = 0; oc < out_channels; oc++) {
        for (int oh = 0; oh < out_height; oh++) {
            for (int ow = 0; ow < out_width; ow++) {
                int32_t acc = 0;

                for (int ic = 0; ic < in_channels; ic++) {
                    for (int kh = 0; kh < kernel_h; kh++) {
                        for (int kw = 0; kw < kernel_w; kw++) {
                            int ih = oh * stride + kh - padding;
                            int iw = ow * stride + kw - padding;

                            if (ih < 0 || ih >= in_height || iw < 0 || iw >= in_width) {
                                continue;
                            }

                            int input_idx = ic * (in_height * in_width) + ih * in_width + iw;
                            int weight_idx = oc * (in_channels * kernel_h * kernel_w)
                                           + ic * (kernel_h * kernel_w)
                                           + kh * kernel_w
                                           + kw;

                            acc += (int32_t)input.data[input_idx] * (int32_t)weights.data[weight_idx];
                        }
                    }
                }

                // Store the accumulator (clamped to INT8 range for now)
                if (acc > 127) acc = 127;
                if (acc < -128) acc = -128;

                int output_idx = oc * (out_height * out_width) + oh * out_width + ow;
                output->data[output_idx] = (int8_t)acc;
            }
        }
    }

    return output;
}