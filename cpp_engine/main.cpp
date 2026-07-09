#include <iostream>
#include "model.h"

int main() {
    std::cout << "Custom AI Inference Engine" << std::endl;
    
    // Load metadata
    Model model = load_model_metadata("../../quantization/model_int8.json");
    std::cout << "Loaded metadata for " << model.conv_layers.size() << " conv layers" << std::endl;
    
    // Load actual weight bytes
    load_model_weights(model, "../../quantization/model_int8.bin");
    
    // Verify by printing layer 0's tensor info
    std::cout << "\nLayer 0 weights:" << std::endl;
    model.conv_layers[0].weights->print_info();
    
    return 0;
}