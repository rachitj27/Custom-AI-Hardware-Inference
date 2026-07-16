#include "ops.h"
#include <stdexcept>
#include <cmath>
#include <cstring>

std::unique_ptr<Tensor> conv2d(
    const Tensor& input,
    const Tensor& weights,
    int stride,
    int padding,
    float input_scale,
    float weight_scale,
    float output_scale,
    int output_zero_point
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
    //main 6 nested for loops to do the convolution
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

            // Compute the combined scale factor
            float M = input_scale * weight_scale / output_scale;
            // Requantization math with no scale factor
            int32_t requantized = (int32_t)std::round(M * acc) + output_zero_point;

            // Clamp to INT8 range just in case of floating point rounding mistakes or calibration values out of range of what quantizatio was done with
            if (requantized > 127) requantized = 127;
            if (requantized < -128) requantized = -128;

            int output_idx = oc * (out_height * out_width) + oh * out_width + ow;
            output->data[output_idx] = (int8_t)requantized;
            }
        }
    }

    return output;
}
void silu_inplace(
    Tensor& tensor,
    float input_scale,
    int input_zero_point,
    float output_scale,
    int output_zero_point
) {
    
    // For each of the 256 possible input values, compute the SiLU result
    int8_t lookup[256];
    for (int i = -128; i <= 127; i++) {
        // Dequantize input value to FP32
        float x_fp32 = input_scale * (i - input_zero_point);
        
        // Apply SiLU: x * sigmoid(x)
        float sigmoid = 1.0f / (1.0f + std::exp(-x_fp32));
        float silu = x_fp32 * sigmoid;
        
        // Requantize to INT8
        int32_t q = (int32_t)std::round(silu / output_scale) + output_zero_point;
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        
        lookup[i + 128] = (int8_t)q;
    }
    
    // Apply lookup to every element in the tensor
    for (size_t i = 0; i < tensor.num_elements; i++) {
        int idx = tensor.data[i] + 128;  // shift to 0-255 range
        tensor.data[i] = lookup[idx];
    }
}
std::unique_ptr<Tensor> upsample2x(const Tensor& input) {
    // Input shape: (channels, height, width)
    // Output shape: (channels, height * 2, width * 2)
    int channels = input.shape[0];
    int in_h = input.shape[1];
    int in_w = input.shape[2];
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    
    auto output = std::make_unique<Tensor>(std::vector<int>{channels, out_h, out_w});
    
    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                // Nearest neighbor: divide by 2 to get input position
                int ih = oh / 2;
                int iw = ow / 2;
                
                int in_idx  = c * (in_h * in_w)  + ih * in_w + iw;
                int out_idx = c * (out_h * out_w) + oh * out_w + ow;
                
                output->data[out_idx] = input.data[in_idx];
            }
        }
    }
    
    return output;
}

std::unique_ptr<Tensor> concat(const std::vector<const Tensor*>& tensors) {
    // All tensors must have same height and width, we concat along channel dim
    if (tensors.empty()) {
        throw std::runtime_error("Concat needs at least one tensor");
    }
    
    // Verify all tensors have the same spatial shape
    int height = tensors[0]->shape[1];
    int width  = tensors[0]->shape[2];
    int total_channels = 0;
    for (const Tensor* t : tensors) {
        if (t->shape[1] != height || t->shape[2] != width) {
            throw std::runtime_error("Concat tensors must have matching spatial dims");
        }
        total_channels += t->shape[0];
    }
    
    auto output = std::make_unique<Tensor>(std::vector<int>{total_channels, height, width});
    
    // Copy each input tensor's data into the output at the right channel offset
    int channel_offset = 0;
    for (const Tensor* t : tensors) {
        int t_channels = t->shape[0];
        size_t bytes_per_channel = height * width;  // INT8 = 1 byte per element
        
        for (int c = 0; c < t_channels; c++) {
            const int8_t* src = t->data + c * bytes_per_channel;
            int8_t* dst = output->data + (channel_offset + c) * bytes_per_channel;
            std::memcpy(dst, src, bytes_per_channel);
        }
        
        channel_offset += t_channels;
    }
    
    return output;
}