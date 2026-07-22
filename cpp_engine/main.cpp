#include <iostream>
#include <fstream>
#include "model.h"
#include "ops.h"

int main() {
    std::cout << "Custom AI Inference Engine" << std::endl;
    
    Model model = load_model_metadata("../../quantization/model_int8.json");
    load_model_weights(model, "../../quantization/model_int8.bin");
    std::cout << "Model loaded" << std::endl;
    
    Tensor input({3, 640, 640});
    std::ifstream input_file("../test_input.bin", std::ios::binary);
    input.load_from_stream(input_file, 3 * 640 * 640);
    std::cout << "Loaded test_input.bin\n" << std::endl;
    
    std::cout << "Running forward pass..." << std::endl;
    auto outputs = run_forward(model, input);
    
    std::cout << "\nForward pass complete" << std::endl;
    std::cout << "Final layer shape: (";
    for (size_t j = 0; j < outputs.back()->shape.size(); j++) {
        std::cout << outputs.back()->shape[j];
        if (j < outputs.back()->shape.size() - 1) std::cout << ", ";
    }
    
    std::cout << ")" << std::endl;
    std::cout << "First 8 output values: ";
    for (int j = 0; j < 8; j++) {
        std::cout << (int)outputs.back()->data[j] << " ";
    }
    std::cout << std::endl;
    
    return 0;
}