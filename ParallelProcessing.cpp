// ParallelProcessing.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>
#include "reduction_example.h"

std::chrono::steady_clock::time_point tStart;
void startClock() {
    tStart = std::chrono::high_resolution_clock::now();
}
double stopClock() { // returns elapsed time in seconds
    auto tEnd = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(tEnd - tStart).count() * 1e-9;
}

int main(int argc, const char* argv[])
{
    
    std::cout << "Hello World!\n";
    int n = 100000000;
    for (int i = 1; i <= 8; i++) {
        startClock();
        ReductionExample::compute(n, i);
        printf("Time taken: %.6fs\n", stopClock());
    }
    
}

