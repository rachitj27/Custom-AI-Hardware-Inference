#include <iostream>
#include "model.h"
#include "ops.h"

int main() {
    std::cout << "Custom AI Inference Engine" << std::endl;
    
    // Load model
    Model model = load_model_metadata("../../quantization/model_int8.json");
    load_model_weights(model, "../../quantization/model_int8.bin");
    std::cout << "Model loaded" << std::endl;
    
    // Create a dummy input tensor of shape (3, 640, 640)
    // Fill with a simple pattern so we can verify the math
    Tensor input({3, 640, 640});
    for (size_t i = 0; i < input.num_elements; i++) {
        input.data[i] = (int8_t)(i % 128);  // repeating 0-127 pattern
    }
    std::cout << "Input tensor created: (3, 640, 640)" << std::endl;
    
    // Run conv2d for layer 0
    // Layer 0: stride=2, padding=1
    std::cout << "\nRunning conv2d on layer 0..." << std::endl;
    // Get scales for layer 0
const Layer& layer0 = model.conv_layers[0];
float input_scale = 1.0f / 255.0f;  // input image scale (RGB pixels normalized)
int input_zero_point = 0;
float output_scale = model.activation_scales[0].scale;
int output_zero_point = model.activation_scales[0].zero_point;

auto output = conv2d(
    input,
    *layer0.weights,
    2, 1,
    input_scale,
    layer0.weight_scale,
    output_scale,
    output_zero_point
);
    
    std::cout << "conv2d complete" << std::endl;
    output->print_info();
    
    return 0;
}