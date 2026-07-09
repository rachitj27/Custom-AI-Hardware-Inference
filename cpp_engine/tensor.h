#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint>
#include <fstream>

class Tensor {
public:
    std::vector<int> shape;
    int8_t* data;
    size_t num_elements;
    
    
    Tensor(const std::vector<int>& shape);
    
    
    ~Tensor();
    
    
    void print_info() const;
    
    
    int8_t at(size_t index) const;

    void load_from_stream(std::ifstream& file, size_t byte_length);
};

#endif