#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <omp.h>
#include <iostream>
// #include <stdio.h>  // stdio functions are used since C++ streams aren't necessarily thread safe
#include <string>
#include "utils.cuh"
#include "instance.cuh"

using namespace std;

int checkCudaErrors() {
    int err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA ERROR %d %s\n", err, cudaGetErrorString((cudaError_t)err));
    return err;
}

// a simple kernel that simply increments each array element by b
__global__ void kernelAddConstant(int* g_a, const int b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    g_a[idx] += b;
}

// a predicate that checks whether each array element is set to its index plus b
int correctResult(int* data, const int n, const int b) {
    for (int i = 0; i < n; i++)
        if (data[i] != i + b) return 0;

    return 1;
}

char* toChar(const std::vector<char>& olig, int l = 10) {
    char* result = new char[l+1];
    for (int i = 0; i < l; i++)
        result[i] = olig[i];
    result[l] = '\0';
    return result;
}

void printSolution(int* solution, int** offsets, const std::vector<std::vector<char>>& oligs, int length, int start = 0) {
    length--; // without last vertex;
    printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\033[15A");
    for (int k = 0; k + 2 < start % 15; k++) printf("\n");
    for (int j = start; j < length; j++) {
        printf("%s\033[10D\033[B", toChar(oligs[solution[j]]));
        for (int k = 0; k < offsets[solution[j]][solution[j + 1]]; k++)
            printf("\033[C");
        if (j > 0 && j % 15 == 0) printf("\033[15A");
        if (j + 1 == length) for (int k = length % 15; k < 15; k++) printf("\n");
    }
    printf("\n\n");
}

int main(int argc, char* argv[]) {

    Utils::projectDirectory = "C:\\Users\\Michal\\Documents\\Projects\\ParallelProcessing";
    std::vector<Instance> instances = Utils::LoadInstances();

    /*for (int i = 0; i < instances.size(); i++) {
        std::cout << instances[i].toString() << "\n";
    }*/

    int i = 0;
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
    /// Display offset matrix
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
    
    /// Simple Tabu Search from one of the vertices

    int tabuCount = 0;
    int tabuId = 0;
    int tabuLimit = 1000; // number of fragments in tabu list
    int tabuFragmentLength = 5; // length of tabu fragments
    std::vector<std::vector<int>> tabuFragments(tabuLimit, std::vector<int>(tabuFragmentLength));
    
    // populating tabu list
    for (int j = 0; j < s/tabuFragmentLength; j++) {
        for (int k = 0; k < tabuFragmentLength; k++) {
            tabuFragments[j][k] = solution[tabuFragmentLength * j + k];
        }
        tabuCount++;
        tabuId++;
    }
    
    // modifying previus solution considering tabu list
    for (int u = 0; u < 10; u++) {

        int* prevSolution = new int[s];
        prevSolution = solution;
        std::vector<int> infringementFragments;
        std::vector<int> infringementId;
        int infringementCount = 0;
        bool* used = new bool[s] {};
        used[solution[0]] = true;
        for (int j = 1; j < s; j++) {
            int bestLimit = 10;
            int* best = new int[bestLimit] {};
            int bestCount = 0;
            for (int k = 0; k < s; k++) {
                if (used[k]) continue;
                int id = bestCount;
                int b = bestCount - 1;
                while (b >= 0 && offsets[solution[j - 1]][k] < offsets[solution[j - 1]][best[b]])
                    id = b--;
                if (id < bestCount || bestCount < bestLimit) {
                    for (int ll = bestCount - 1; ll > id; ll--)
                        best[ll] = best[ll - 1];
                    best[id] = k;
                    if (bestCount < bestLimit) bestCount++;
                }
            }
            //for (int k = 0; k < bestCount; k++) {
            //    printf("%s --%d--> %s\n", toChar(instances[0].oligs[solution[j - 1]]), offsets[solution[j - 1]][best[k]], toChar(instances[0].oligs[best[k]]));
            //}
            // Check the tabu lists
            int seed = solution[((int)(1000 * sqrt(109 + j * j))) % s];
            // if (seed % 10 == 0) printf("X"); else printf("_"); // does look pretty randomized
            bool chosen = false;
            for (int k = 0; k < bestCount; k++) {
                int probModulo = 0;
                for (int kk = 0; kk < infringementCount; kk++) {
                    infringementId[kk] += (infringementId[kk] < tabuFragmentLength - 1) ? 1 : 0;
                    if (tabuFragments[infringementFragments[kk]][infringementId[kk]] == best[k]) {
                        probModulo = tabuFragmentLength - infringementId[kk];
                        if (seed % probModulo == 0) {
                            solution[j] = best[k];
                            used[best[k]] = true;
                        }
                        break;
                    }
                }
                if (probModulo == 0 && infringementCount > 0) {
                    infringementFragments.clear();
                    infringementId.clear();
                    infringementCount = 0;
                }

                if (seed % tabuFragmentLength == 0) {
                    solution[j] = best[k];
                    used[best[k]] = true;
                }
                // adding new infringements
                bool infringing = false;
                for (int kk = 0; kk < tabuCount; kk++)
                    if (tabuFragments[kk][0] == best[k]) {
                        if (used[best[k]]) {
                            infringementFragments.push_back(k);
                            infringementId.push_back(0);
                            infringementCount++;
                        }
                        else {
                            infringing = true;
                            break;
                        }
                    }
                if (!infringing) {
                    solution[j] = best[k];
                    used[best[k]] = true;
                }
                if (used[best[k]]) {
                    printf("%d", k);
                    chosen = true;
                    break;
                }
            }
            if (!chosen) {
                // fallback (greedy):
                solution[j] = best[0];
                used[best[0]] = true;
            }
        }
        // updating the tabu list
        printf("TabuCount: %d\n", tabuCount);
        for (int j = 0; j < s / tabuFragmentLength; j++) {
            for (int k = 0; k < tabuFragmentLength; k++) {
                printf("_%d", tabuCount);
                tabuFragments[tabuId++ % tabuCount][k] = solution[tabuFragmentLength * j + k];
            }
            if (tabuCount < tabuLimit) tabuCount++;
        }
        // evaluating the solution
        for (int j = 0; j < s; j++) {

        }

        printf("Solution for %s: (press any key to continue)\n", toChar(instances[0].oligs[solution[0]]));
        getchar();
        printSolution(solution, offsets, instances[0].oligs, 100);
        printf("\t\t\t . . .\r\n\n\n");
        printSolution(solution, offsets, instances[0].oligs, s / 2 + 80, s / 2);
        printf("\t\t\t . . .\r\n\n\n");
        printSolution(solution, offsets, instances[0].oligs, s, s - 30);
        printf("\r\n\n\n");

    }

    /// Cleanup
    for (int j = 0; j < s; ++j) { delete[] greedy[j]; }
    delete[] greedy;
    for (int j = 0; j < s; ++j) { delete[] offsets[j]; }
    delete[] offsets;

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

    /////////////////////////////////////////////////////////////////
    // display CPU and GPU configuration
    //
    printf("number of host CPUs:\t%d\n", omp_get_num_procs());
    printf("number of CUDA devices:\t%d\n", num_gpus);

    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp dprop;
        cudaGetDeviceProperties(&dprop, i);
        printf("   %d: %s\n", i, dprop.name);
    }

    printf("---------------------------\n");

    /////////////////////////////////////////////////////////////////
    // initialize data
    //
    unsigned int n = num_gpus * 8192;
    unsigned int nbytes = n * sizeof(int);
    int* a = 0;  // pointer to data on the CPU
    int b = 3;   // value by which the array is incremented
    a = (int*)malloc(nbytes);

    if (0 == a) {
        printf("couldn't allocate CPU memory\n");
        return 1;
    }

    for (unsigned int i = 0; i < n; i++) a[i] = i;

    ////////////////////////////////////////////////////////////////
    // run as many CPU threads as there are CUDA devices
    //   each CPU thread controls a different device, processing its
    //   portion of the data.  It's possible to use more CPU threads
    //   than there are CUDA devices, in which case several CPU
    //   threads will be allocating resources and launching kernels
    //   on the same device.  For example, try omp_set_num_threads(2*num_gpus);
    //   Recall that all variables declared inside an "omp parallel" scope are
    //   local to each CPU thread
    //
omp_set_num_threads(num_gpus);  // create as many CPU threads as there are CUDA devices
//omp_set_num_threads(2*num_gpus);// create twice as many CPU threads as there
  // are CUDA devices
#pragma omp parallel
    {
        unsigned int cpu_thread_id = omp_get_thread_num();
        unsigned int num_cpu_threads = omp_get_num_threads();

        // set and check the CUDA device for this CPU thread
        int gpu_id = -1;
        cudaSetDevice(
            cpu_thread_id %
            num_gpus);
        cudaGetDevice(&gpu_id);
        printf("CPU thread %d (of %d) uses CUDA device %d\n", cpu_thread_id,
            num_cpu_threads, gpu_id);

        int* d_a = 0;  // pointer to memory on the device associated with this CPU thread
        int* sub_a = a + cpu_thread_id * n / num_cpu_threads;  // pointer to this CPU thread's portion of data
        unsigned int nbytes_per_kernel = nbytes / num_cpu_threads;
        dim3 gpu_threads(128);  // 128 threads per block
        dim3 gpu_blocks(n / (gpu_threads.x * num_cpu_threads));

        /*for (int i = 0; i < n; i++)
            printf("%d\t", a[i]);*/
        cudaMalloc((void**)&d_a, nbytes_per_kernel);

        cudaMemset(d_a, 0, nbytes_per_kernel);
        cudaMemcpy(d_a, a, nbytes_per_kernel, cudaMemcpyHostToDevice);
        checkCudaErrors();
        kernelAddConstant<<<gpu_blocks, gpu_threads >>>(d_a, b);
        checkCudaErrors();
        cudaMemcpy(sub_a, d_a, nbytes_per_kernel, cudaMemcpyDeviceToHost);//);
        cudaFree(d_a);
    }
    printf("---------------------------\n");

    if (cudaSuccess != cudaGetLastError())
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    else
        printf("code executed without errors");

    ////////////////////////////////////////////////////////////////
    // check the result
    //
    bool bResult = correctResult(a, n, b);

    //for (int i = 0; i < n; i++)
    //    printf("%d\t", a[i]);
    if (a) free(a);  // free CPU memory

    exit(bResult ? EXIT_SUCCESS : EXIT_FAILURE);
}