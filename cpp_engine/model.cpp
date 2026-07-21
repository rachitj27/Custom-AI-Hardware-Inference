#include "model.h"
#include <fstream>
#include <iostream>
#include <utility>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

Model load_model_metadata(const std::string& json_path) {
    // Open and parse JSON
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + json_path);
    }
    
    json metadata;
    file >> metadata;
    
    Model model;
    model.num_bits = metadata["num_bits"];
    
    // Parse each convulution layer
    for (const auto& layer_json : metadata["conv_layers"]) {
        Layer layer;
        layer.layer_id = layer_json["layer_id"];
        layer.path = layer_json["path"];
        layer.weight_shape = layer_json["weight_shape"].get<std::vector<int>>();
        layer.byte_offset = layer_json["byte_offset"];
        layer.byte_length = layer_json["byte_length"];
        layer.weight_scale = layer_json["weight_scale"];
        layer.weight_zero_point = layer_json["weight_zero_point"];
        layer.quantization_scheme = layer_json["quantization_scheme"];
        
        model.conv_layers.push_back(std::move(layer));
    }
    
    // Parse activation scales
    for (auto it = metadata["activation_scales"].begin(); it != metadata["activation_scales"].end(); ++it) {
        int layer_idx = std::stoi(it.key());
        ActivationScale act;
        act.scale = (*it)["scale"];
        act.zero_point = (*it)["zero_point"];
        model.activation_scales[layer_idx] = act;
    }
    // Parse architecture
    if (metadata.contains("architecture")) {
        for (const auto& arch_json : metadata["architecture"]) {
            ArchLayer arch;
            arch.layer_id = arch_json["layer_id"];
            arch.type = arch_json["type"];
            arch.conv_ids = arch_json["conv_ids"].get<std::vector<int>>();
            
            // Handle input_from: could be int or array
            const auto& input_from = arch_json["input_from"];
            if (input_from.is_array()) {
                arch.is_multi_input = true;
                arch.input_from_multi = input_from.get<std::vector<int>>();
                arch.input_from_single = -1;
            } else if (input_from.is_string()) {
                // "input" — the very first layer
                arch.is_multi_input = false;
                arch.input_from_single = -1;  // -1 signals raw input
            } else {
                arch.is_multi_input = false;
                arch.input_from_single = input_from.get<int>();
            }
            
            // Extra fields with defaults
            arch.n_bottlenecks = arch_json.value("n", 0);
            arch.shortcut = arch_json.value("shortcut", false);
            arch.kernel_size = arch_json.value("k", 5);
            
            model.architecture.push_back(arch);
        }
    }
    return model;
}

void load_model_weights(Model& model, const std::string& bin_path) {
    std::ifstream file(bin_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open " + bin_path);
    }
    
    for (auto& layer : model.conv_layers) {
        // Create the tensor with the layer's shape
        layer.weights = std::make_unique<Tensor>(layer.weight_shape);
        
        
        file.seekg(layer.byte_offset);
        
        // Read the bytes into the tensor
        layer.weights->load_from_stream(file, layer.byte_length);
    }
    
    std::cout << "Loaded weights for " << model.conv_layers.size() << " conv layers" << std::endl;
}