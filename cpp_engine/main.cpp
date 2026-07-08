#include <iostream>
#include "model.h"

int main() {
    std::cout << "Custom AI Inference Engine" << std::endl;
    
    Model model = load_model_metadata("../../quantization/model_int8.json");
    
    std::cout << "Loaded model" << std::endl;
    std::cout << "  num_bits: " << model.num_bits << std::endl;
    std::cout << "  num conv layers: " << model.conv_layers.size() << std::endl;
    std::cout << "  num activation scales: " << model.activation_scales.size() << std::endl;
    
    // Print first few layers
    std::cout << "\nFirst 3 conv layers:" << std::endl;
    for (int i = 0; i < 3; i++) {
        const Layer& l = model.conv_layers[i];
        std::cout << "  Layer " << l.layer_id << " (" << l.path << ")" << std::endl;
        std::cout << "    shape: (";
        for (size_t j = 0; j < l.weight_shape.size(); j++) {
            std::cout << l.weight_shape[j];
            if (j < l.weight_shape.size() - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
        std::cout << "    scale: " << l.weight_scale << ", zero_point: " << l.weight_zero_point << std::endl;
        std::cout << "    byte_offset: " << l.byte_offset << ", byte_length: " << l.byte_length << std::endl;
    }
    
    return 0;
}