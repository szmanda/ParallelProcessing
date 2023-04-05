// ParallelProcessing.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <chrono>
#include "reduction_example.h"
#include "pi_example.h"

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
    
    std::cout << "Regression Example\n";
    int n = 1000000000;
    for (int i = 1; i <= 8; i++) {
        startClock();
        int result = ReductionExample::compute(n, i);
        double elapsed = stopClock();
        printf("\t%d cores, result: %d, time taken: %.6fs\n", i, result, elapsed);
    }
    std::cout << "Calculating PI\n";
    std::cout << "#Cores\tResult_1e8\tTime[s]_1e8\tTime[s]_1e9\tTime[s]_1e10\n";
    // number of cores
    for (int i = 1; i <= 8; i++) {
        n = 1e8;
        double result, elapsed;
        startClock();
        result = PiExample::compute(n, i);
        elapsed = stopClock();
        printf("%d\t%.9f\t%.8f\t", i, result, elapsed);

        n = 1e9;
        startClock();
        PiExample::compute(n, i);
        elapsed = stopClock();
        printf("%.8f\t", elapsed);
        
        n = 1e10;
        startClock();
        PiExample::compute(n, i);
        elapsed = stopClock();
        printf("%.8f\n", elapsed);
    }
    
}

