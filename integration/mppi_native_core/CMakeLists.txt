cmake_minimum_required(VERSION 3.16)
project(ranger_mppi_native_core LANGUAGES CXX)
add_library(ranger_mppi_native SHARED core.cpp)
target_compile_features(ranger_mppi_native PRIVATE cxx_std_17)
target_compile_options(ranger_mppi_native PRIVATE -O3 -Wall -Wextra -Wpedantic)
# Deliberately no fast-math: preserve finite-rate steering and cost semantics.
