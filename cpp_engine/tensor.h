#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <cstdint>

class Tensor {
public:
    std::vector<int> shape;
    int8_t* data;
    size_t num_elements;
    
    
    Tensor(const std::vector<int>& shape);
    
    
    ~Tensor();
    
    
    void print_info() const;
    
    
    int8_t at(size_t index) const;
};

#endif