#include <iostream>
#include "tensor.h"

int main() {
    std::cout << "Custom AI Inference Engine" << std::endl;
    
    // test tensor
    Tensor t({16, 3, 3, 3});
    
    // test values
    for (size_t i = 0; i < t.num_elements; i++) {
        t.data[i] = (int8_t)(i % 128);
    }
    
    
    t.print_info();
    
    return 0;
}