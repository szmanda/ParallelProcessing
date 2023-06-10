#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>
#include <iostream>
// #include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe
#include <string>
#include "utils.cuh"
#include "instance.cuh"


__host__ __device__ char* toChar(const std::vector<char>& olig, int l = 10);
__host__ void printSolution(int* solution, int** offsets, const std::vector<std::vector<char>>& oligs, int length, int start = 0);
__device__ void printSolution(int* solution, int* offsets_flat, int s, const std::vector<std::vector<char>>& oligs, int length, int start = 0);

__global__ void kernelTabuSearch(int* solution, int *offsets, const int s, char* oligs_flat);