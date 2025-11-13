#!/bin/bash

set -e
PYBIND_INCLUDES=$(pybind11-config --includes)
CUDA_INCLUDES=-I$CUDA_PATH/include
CUDA_LINK=$CUDA_PATH/lib/stubs/libcuda.so
CXXFLAGS="-O3 -Wall -shared -fPIC -std=c++17"
g++ $CXXFLAGS $PYBIND_INCLUDES $CUDA_INCLUDES -o cuda_core_holders_demo.so cuda_core_holders_demo.cpp $CUDA_LINK
