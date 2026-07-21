#include <iostream>
#include "model.h"

int main() {
    std::cout << "Custom AI Inference Engine" << std::endl;
    
    Model model = load_model_metadata("../../quantization/model_int8.json");
    load_model_weights(model, "../../quantization/model_int8.bin");
    std::cout << "Model loaded" << std::endl;
    
    std::cout << "\nArchitecture map:" << std::endl;
    for (const auto& arch : model.architecture) {
        std::cout << "  Layer " << arch.layer_id << ": " << arch.type;
        std::cout << " (convs: [";
        for (size_t i = 0; i < arch.conv_ids.size(); i++) {
            std::cout << arch.conv_ids[i];
            if (i < arch.conv_ids.size() - 1) std::cout << ", ";
        }
        std::cout << "], from: ";
        if (arch.is_multi_input) {
            std::cout << "[";
            for (size_t i = 0; i < arch.input_from_multi.size(); i++) {
                std::cout << arch.input_from_multi[i];
                if (i < arch.input_from_multi.size() - 1) std::cout << ", ";
            }
            std::cout << "]";
        } else if (arch.input_from_single == -1) {
            std::cout << "input";
        } else {
            std::cout << arch.input_from_single;
        }
        std::cout << ")" << std::endl;
    }
    
    return 0;
}