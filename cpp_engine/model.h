#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <cstdint>
#include <map>
#include <memory>
#include "tensor.h"
struct ArchLayer {
    int layer_id;
    std::string type;
    std::vector<int> conv_ids;
    
    // input_from can be either a single int or a list of ints (for Concat/Detect)
    bool is_multi_input;
    int input_from_single;         // used when is_multi_input is false
    std::vector<int> input_from_multi;  // used when is_multi_input is true
    
    // Extra fields for specific types
    int n_bottlenecks;             // for C2f, 0 otherwise
    bool shortcut;                 // for C2f
    int kernel_size;               // for SPPF (default 5)
};
struct Layer {
    int layer_id;
    std::string path;
    std::vector<int> weight_shape;
    size_t byte_offset;
    size_t byte_length;
    float weight_scale;
    int weight_zero_point;
    std::string quantization_scheme;
    std::unique_ptr<Tensor> weights;
};

struct ActivationScale {
    float scale;
    int zero_point;
};

struct Model {
    int num_bits;
    std::vector<Layer> conv_layers;
    std::map<int, ActivationScale> activation_scales;
    std::vector<ArchLayer> architecture;
};

// Load model metadata from a JSON file
Model load_model_metadata(const std::string& json_path);
void load_model_weights(Model& model, const std::string& bin_path);

#endif