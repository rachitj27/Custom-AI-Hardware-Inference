#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <vector>
#include <cstdint>
#include <map>
#include <memory>
#include "tensor.h"

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
};

// Load model metadata from a JSON file
Model load_model_metadata(const std::string& json_path);
void load_model_weights(Model& model, const std::string& bin_path);

#endif