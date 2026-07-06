cmake_minimum_required(VERSION 3.15)
project(custom_engine CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-Wall -Wextra -O2)

add_executable(custom_engine main.cpp)