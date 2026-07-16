#include <iostream>
#include <fstream>
#include <cmath>
#include "model.h"
#include "ops.h"

int main() {
    std::cout << "Custom AI Inference Engine" << std::endl;
    
    // Load model
    Model model = load_model_metadata("../../quantization/model_int8.json");
    load_model_weights(model, "../../quantization/model_int8.bin");
    std::cout << "Model loaded" << std::endl;
    
    // Load test input from Python reference
    Tensor input({3, 640, 640});
    std::ifstream input_file("../test_input.bin", std::ios::binary);
    if (!input_file.is_open()) {
        throw std::runtime_error("Could not open test_input.bin");
    }
    input.load_from_stream(input_file, 3 * 640 * 640);
    std::cout << "Loaded test_input.bin" << std::endl;
    std::cout << "  First values: ";
    for (int i = 0; i < 8; i++) std::cout << (int)input.data[i] << " ";
    std::cout << std::endl;
    
    // Run conv2d for layer 0
    const Layer& layer0 = model.conv_layers[0];
    float input_scale = 1.0f / 255.0f;
    float output_scale = model.activation_scales[0].scale;
    int output_zero_point = model.activation_scales[0].zero_point;
    
    std::cout << "\nRunning conv2d on layer 0..." << std::endl;
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
    std::cout << "  First values: ";
    for (int i = 0; i < 8; i++) std::cout << (int)output->data[i] << " ";
    std::cout << std::endl;
    
    // Load reference output
    Tensor reference({16, 320, 320});
    std::ifstream ref_file("../test_output.bin", std::ios::binary);
    if (!ref_file.is_open()) {
        throw std::runtime_error("Could not open test_output.bin");
    }
    reference.load_from_stream(ref_file, 16 * 320 * 320);
    std::cout << "\nLoaded reference output" << std::endl;
    std::cout << "  First values: ";
    for (int i = 0; i < 8; i++) std::cout << (int)reference.data[i] << " ";
    std::cout << std::endl;
    
    // Compare
    int exact_matches = 0;
    int close_matches = 0;  // within 1
    int64_t total_error = 0;
    int max_error = 0;
    
    for (size_t i = 0; i < output->num_elements; i++) {
        int diff = (int)output->data[i] - (int)reference.data[i];
        int abs_diff = diff < 0 ? -diff : diff;
        
        if (diff == 0) exact_matches++;
        if (abs_diff <= 1) close_matches++;
        total_error += abs_diff;
        if (abs_diff > max_error) max_error = abs_diff;
    }
    
    double avg_error = (double)total_error / output->num_elements;
    double exact_pct = 100.0 * exact_matches / output->num_elements;
    double close_pct = 100.0 * close_matches / output->num_elements;
    
    std::cout << "\n=== Comparison Results ===" << std::endl;
    std::cout << "Total elements: " << output->num_elements << std::endl;
    std::cout << "Exact matches:  " << exact_matches << " (" << exact_pct << "%)" << std::endl;
    std::cout << "Close (±1):     " << close_matches << " (" << close_pct << "%)" << std::endl;
    std::cout << "Average error:  " << avg_error << std::endl;
    std::cout << "Max error:      " << max_error << std::endl;
    
    return 0;
}