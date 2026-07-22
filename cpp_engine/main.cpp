#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include "model.h"
#include "ops.h"

int main() {
    std::cout << "Custom AI Inference Engine" << std::endl;
    
    Model model = load_model_metadata("../../quantization/model_int8.json");
    load_model_weights(model, "../../quantization/model_int8.bin");
    std::cout << "Model loaded" << std::endl;
    
    Tensor input({3, 640, 640});
    std::ifstream input_file("../test_input.bin", std::ios::binary);
    input.load_from_stream(input_file, 3 * 640 * 640);
    std::cout << "Loaded test_input.bin\n" << std::endl;
    
    // Warm-up run
    std::cout << "Warming up..." << std::endl;
    auto warm = run_forward(model, input);
    
    // Timed runs
    const int n_runs = 5;
    std::vector<double> times;
    std::cout << "\nRunning " << n_runs << " timed inferences..." << std::endl;
    
    for (int i = 0; i < n_runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        auto outputs = run_forward(model, input);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
        std::cout << "  Run " << (i+1) << ": " << ms << " ms" << std::endl;
    }
    
    // Stats
    double sum = 0, minv = times[0], maxv = times[0];
    for (double t : times) {
        sum += t;
        if (t < minv) minv = t;
        if (t > maxv) maxv = t;
    }
    double avg = sum / n_runs;
    
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << "Avg: " << avg << " ms" << std::endl;
    std::cout << "Min: " << minv << " ms" << std::endl;
    std::cout << "Max: " << maxv << " ms" << std::endl;
    
    return 0;
}