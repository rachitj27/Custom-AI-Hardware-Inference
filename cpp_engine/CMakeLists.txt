cmake_minimum_required(VERSION 3.15)
project(custom_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -O2)

# Find nlohmann/json
find_package(nlohmann_json 3.2.0 REQUIRED)

add_executable(custom_engine main.cpp tensor.cpp model.cpp ops.cpp)
target_link_libraries(custom_engine PRIVATE nlohmann_json::nlohmann_json)