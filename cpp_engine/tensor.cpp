#include "tensor.h"
#include <iostream>
#include <stdexcept>

// Constructor
Tensor::Tensor(const std::vector<int>& shape_in) {
    shape = shape_in;
    
    // Compute total number of elements by multiplying all dimensions
    num_elements = 1;
    for (int dim : shape) {
        num_elements *= dim;
    }
    
    // Allocate the buffer
    data = new int8_t[num_elements];
}

// Destructor
Tensor::~Tensor() {
    delete[] data;
}

// Print shape and first 8 values
void Tensor::print_info() const {
    std::cout << "Tensor shape: (";
    for (size_t i = 0; i < shape.size(); i++) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << ")" << std::endl;
    std::cout << "Num elements: " << num_elements << std::endl;
    
    std::cout << "First values: ";
    size_t n_to_print = std::min(num_elements, (size_t)8);
    for (size_t i = 0; i < n_to_print; i++) {
        std::cout << (int)data[i] << " ";
    }
    std::cout << std::endl;
}

// Get value at flat index
int8_t Tensor::at(size_t index) const {
    return data[index];
}
void Tensor::load_from_stream(std::ifstream& file, size_t byte_length) {
    if (byte_length != num_elements) {
        throw std::runtime_error("Byte length does not match tensor size");
    }
    file.read(reinterpret_cast<char*>(data), byte_length);
    if (!file) {
        throw std::runtime_error("Failed to read from file");
    }
}