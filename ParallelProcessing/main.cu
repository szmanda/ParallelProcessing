#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>
#include <iostream>
// #include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe
#include <string>
#include "utils.cuh"
#include "instance.cuh"
#include "kernel.cuh"

using namespace std;

int checkCudaErrors() {
    int err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA ERROR %d %s\n", err, cudaGetErrorString((cudaError_t)err));
    return err;
}

//char* toChar(const std::vector<char>& olig, int l = 10) {
//    char* result = new char[l+1];
//    for (int i = 0; i < l; i++)
//        result[i] = olig[i];
//    result[l] = '\0';
//    return result;
//}
//
//void printSolution(int* solution, int** offsets, const std::vector<std::vector<char>>& oligs, int length, int start = 0) {
//    length--; // without last vertex;
//    printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\033[15A");
//    for (int k = 0; k + 2 < start % 15; k++) printf("\n");
//    for (int j = start; j < length; j++) {
//        printf("%s\033[10D\033[B", toChar(oligs[solution[j]]));
//        for (int k = 0; k < offsets[solution[j]][solution[j + 1]]; k++)
//            printf("\033[C");
//        if (j > 0 && j % 15 == 0) printf("\033[15A");
//        if (j + 1 == length) for (int k = length % 15; k < 15; k++) printf("\n");
//    }
//    printf("\n\n");
//}

//int benchmark();
////int run(int argc, char* argv[]);
//
//int ma(int argc, char* argv[]) {
//    freopen("NUL", "w", stdout);
//    if (argc > 1 && std::string(argv[1]) == "benchmark") return benchmark();
//    //else return run(argc, argv);
//    return 0;
//}

int benchmark() {
    std::vector<int> blocks = { 8, 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    std::vector<int> threads = { 64, 1, 2, 4, 8, 16, 32, 64, 128, 256 };
    std::vector<int> iterations = { 1 };

    int total_c = blocks.size() * threads.size() * iterations.size();
    int c = 0;

    std::cout << "total:" << total_c << std::endl;
    std::cout << "blocks,threads,iterations,time,completed" << std::endl;

    for (int b : blocks) {
        for (int t : threads) {
            for (int i : iterations) {
                if (c < 5) { c++; continue; }
                char argv1[10];
                char argv2[10];
                char argv3[10];
                char argv4[10];
                snprintf(argv1, sizeof(argv1), "%d", b);
                snprintf(argv2, sizeof(argv2), "%d", t);
                snprintf(argv3, sizeof(argv3), "%d", i);
                snprintf(argv4, sizeof(argv4), "%d", 0);
                char* argv[] = { "benchmark", argv1, argv2, argv3, argv4 };
                int argc = 5;

                freopen("NUL", "w", stdout);

                cudaEvent_t start, stop;
                cudaEventCreate(&start);
                cudaEventCreate(&stop);
                    cudaEventRecord(start, 0);
                    //run(argc, argv);
                    cudaEventRecord(stop, 0);
                cudaEventSynchronize(stop);

                float milliseconds = 0;
                cudaEventElapsedTime(&milliseconds, start, stop);
                float seconds = milliseconds / 1000.0;

                freopen("CON", "w", stdout);

                // Print the execution time
                std::cout << b << "," << t << "," << i << "," << seconds << "," << c + 1 << "/" << total_c << std::endl;

                // Destroy CUDA events
                cudaEventDestroy(start);
                cudaEventDestroy(stop);
                checkCudaErrors();
                cudaDeviceReset();

                c++;

                if (checkCudaErrors()) return 1;
            }
        }
    }
    return 0;
}


int main(int argc, char* argv[]) {
    freopen("NUL", "w", stdout);
    int param_blocks = 4;
    int param_threads = 64;
    int param_iterations = 1;
    int param_instance_id = 0;
    if (argc >= 2) param_blocks = std::stoi(argv[1]);
    if (argc >= 3) param_threads = std::stoi(argv[2]);
    if (argc >= 4) param_iterations = std::stoi(argv[3]);
    if (argc >= 5) param_instance_id = std::stoi(argv[4]);

    Utils::projectDirectory = "C:\\Users\\Michal\\Documents\\Projects\\ParallelProcessing";
    std::vector<Instance> instances = Utils::LoadInstances();

    /*for (int i = 0; i < instances.size(); i++) {
        std::cout << instances[i].toString() << "\n";
    }*/

    int i = param_instance_id;
    instances[i].oligs;
    int s = instances[i].s;
    int l = instances[i].l;
    int** offsets = new int* [s];
    for (int j = 0; j < s; j++) { offsets[j] = new int[s]; }
    // Calculate offset matrix
#pragma omp parallel for
    for (int o1 = 0; o1 < s; o1++) {
        // std::cout << toChar(instances[i].oligs[o1]) << std::endl;
        for (int o2 = 0; o2 < s; o2++) {
            offsets[o1][o2] = 10;
            for (int ii = 0; ii < l; ii++) {
                bool matchPositive = true;
                for (int jj = 0; jj < l - ii; jj++) {
                    if (instances[i].oligs[o1][ii + jj] != instances[i].oligs[o2][jj]) {
                        matchPositive = false;
                    }
                }
                if (matchPositive) {
                    offsets[o1][o2] = ii;
                    break;
                }
            }
        }
    }
    char* oligs_flat = new char[s * l];
    for (int ii = 0; ii < s; ii++) {
        char* olig = toChar(instances[i].oligs[ii]);
        for (int j = 0; j < l; j++) {
            oligs_flat[ii * l + j] = olig[j];
        }
    }
    int* offsets_flat = new int[s * s];
    for (int ii = 0; ii < s; ii++) {
        for (int j = 0; j < s; j++) {
            offsets_flat[ii * s + j] = offsets[ii][j];
        }
    }
    /// Print offset matrix
    /*for (int o1 = 0; o1 < s; o1++) {
        std::cout << toChar(instances[i].oligs[o1]) << "\t";
    }
    for (int o1 = 0; o1 < s; o1++) {
        std::cout << toChar(instances[i].oligs[o1]);
        for (int o2 = 0; o2 < s; o2++) {
            std::cout << offsets[o1][o2] << "\t\t";
        }
        std::cout << "\n";
    }*/
    
    /// Finding greedy solutions
    int** greedy = new int* [s];
    for (int j = 0; j < s; j++) { greedy[j] = new int[s]; }
#pragma omp parallel for
    for (int greedyId = 0; greedyId < s; greedyId++) {
        int* solution = new int[s];
        solution[0] = greedyId;
        for (int j = 1; j < s; j++) {
            int bestUnused;
            int bestUnusedOffset = 100;
            for (int k = 0; k < s; k++) {
                if (offsets[solution[j - 1]][k] < bestUnusedOffset) {
                    int repeat = 0;
                    while (repeat < j) {
                        if (solution[repeat] == k) break;
                        repeat++;
                    }
                    if (solution[repeat] != k) {
                        bestUnused = k;
                        bestUnusedOffset = offsets[solution[j - 1]][k];
                    }
                }
            }
            solution[j] = bestUnused;
        }
        greedy[greedyId] = solution;
    }
    /// Print out the greedy solution
    int* solution = new int[s];
    solution = greedy[254];
    printf("Greedy solution for %s: (press any key to continue)\n", toChar(instances[0].oligs[solution[0]]));
    // getchar();
    printSolution(solution, offsets, instances[0].oligs, 100);
    printf("\t\t\t . . .\n\n\n");
    printSolution(solution, offsets, instances[0].oligs, s/2+80, s/2);
    printf("\t\t\t . . .\n\n\n");
    printSolution(solution, offsets, instances[0].oligs, s, s - 30);
    printf("\r\n\n\n");
    
    // Tabu Search

    /// Cleanup
    /*for (int j = 0; j < s; ++j) { delete[] greedy[j]; }
    delete[] greedy;
    for (int j = 0; j < s; ++j) { delete[] offsets[j]; }
    delete[] offsets;*/

    printf("\n\n\n");
    /////////////////////////////////////////////////////////////////
    // CUDA Toolkit version
    //
    int num_gpus = 0;  // number of CUDA GPUs
    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    int major = runtimeVersion / 1000;
    int minor = (runtimeVersion % 100) / 10;
    printf("CUDA Toolkit Version: %d.%d\n", major, minor);


    printf("%s Starting...\n\n", argv[0]);

    /////////////////////////////////////////////////////////////////
    // determine the number of CUDA capable GPUs
    //
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus < 1) {
        printf("no CUDA capable devices were detected\n");
        return 1;
    }

    ///// Running Tabu search on CUDA

    int* d_solution = nullptr;
    int* d_offsets = nullptr; // will store a flattened version of offsets
    char* d_oligs_flat = nullptr;
    int* d_tabu_fragments = nullptr;
    int* d_tabu_count = nullptr;
    int* d_tabu_id = nullptr;
    bool* d_used = nullptr;

    //int* sub_a = a + cpu_thread_id * n / num_cpu_threads;  // pointer to this CPU thread's portion of data

    int local_tabu_limit = 2000;
    int local_tabu_fragment_length = 5;


    dim3 threads(param_threads);  // number of threads per block
    dim3 blocks(param_blocks);
    unsigned int solution_size = blocks.x * threads.x * s * sizeof(int);
    unsigned int offsets_size = s * s * sizeof(int);
    unsigned int oligs_flat_size = s * l * sizeof(char);
    unsigned int tabu_fragments_size = blocks.x * threads.x * local_tabu_limit * local_tabu_fragment_length * sizeof(int);
    unsigned int tabu_count_size = sizeof(int);
    unsigned int tabu_id_size = sizeof(int);
    unsigned int used_size = blocks.x * threads.x * s * sizeof(bool);

    printf("Total memory usage: %d", solution_size + offsets_size + oligs_flat_size + tabu_fragments_size
        + tabu_count_size + tabu_id_size + used_size);

    /*for (int i = 0; i < n; i++)
        printf("%d\t", a[i]);*/
    printf("malloc\n");
    cudaMalloc((void**)&d_solution, solution_size);
    cudaMalloc((void**)&d_offsets, offsets_size);
    cudaMalloc((void**)&d_oligs_flat, oligs_flat_size);
    cudaMalloc((void**)&d_tabu_fragments, tabu_fragments_size);
    cudaMalloc((void**)&d_tabu_count, tabu_count_size);
    cudaMalloc((void**)&d_tabu_id, tabu_id_size);
    cudaMalloc((void**)&d_used, used_size);
    checkCudaErrors();

    printf("memset\n");
    cudaMemset(d_tabu_count, 0, tabu_count_size);
    cudaMemset(d_tabu_id, 0, tabu_id_size);
    checkCudaErrors();
    
    printf("memcopy\n");
    /*for (int j = 0; j < blocks.x * threads.x; j++) {
        cudaMemcpy(&d_solution[j * s], greedy[j], s * sizeof(int), cudaMemcpyHostToDevice);
    }*/
    cudaMemcpy(d_offsets, offsets_flat, offsets_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_oligs_flat, oligs_flat, oligs_flat_size, cudaMemcpyHostToDevice);
    checkCudaErrors();
    
    int batches = s / (blocks.x * threads.x); // full batches - ignoring the potential remainder
    for (int j = 0; j < batches; j++) {
        if (j == batches) { // last batch (remainder)
            threads = dim3(s % (blocks.x * threads.x));
            blocks = dim3(1);
            if (threads.x == 0) break;
            printf("Last batch: handling the remainder of the batch division\n");
        }
        printf("memcopy %d/%d batch\n", j + 1, batches);
        for (int k = 0; k < blocks.x * threads.x; k++) {
            cudaMemcpy(&d_solution[k * s], greedy[j], k * sizeof(int), cudaMemcpyHostToDevice);
        }
        for (int k = 0; k < param_iterations; k++) {
            cudaMemset(d_used, 0, used_size);
            checkCudaErrors();
            printf("Starting kernel %d/%d\n", k + 1, param_iterations);
            kernelTabuSearch <<< blocks, threads >>> (
                d_solution,
                d_offsets,
                s,
                instances[i].n,
                d_oligs_flat,
                d_tabu_fragments,
                local_tabu_limit,
                local_tabu_fragment_length,
                d_tabu_count,
                d_tabu_id,
                d_used
                );
            cudaDeviceSynchronize();
            printf("kernel %d/%d completed.\n", k + 1, param_iterations);
        }
        if (checkCudaErrors()) continue;
        printf("Finished  %d/%d batch\n", j + 1, batches);
        printf("Copied first solution back to host:\n\n");
        cudaMemcpy(solution, d_solution, s * sizeof(int), cudaMemcpyDeviceToHost);
        checkCudaErrors();
        printSolution(solution, offsets, instances[0].oligs, 100);
        printf("\t\t\t . . .\n\n\n");
        printSolution(solution, offsets, instances[0].oligs, s / 2 + 80, s / 2);
        printf("\t\t\t . . .\n\n\n");
        printSolution(solution, offsets, instances[0].oligs, s, s - 30);
        printf("\r\n\n\n");
    }
    checkCudaErrors();
    cudaFree(d_solution);
    cudaFree(d_offsets);
    cudaFree(d_oligs_flat);
    cudaFree(d_tabu_fragments);
    cudaFree(d_tabu_count);
    cudaFree(d_tabu_id);
    cudaFree(d_used);
    checkCudaErrors();

    printf("Finished!");
    
    return 0;
}