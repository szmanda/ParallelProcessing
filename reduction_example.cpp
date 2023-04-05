/*
Kompilacja:
  $ gcc ompfor1.c -o ompfor1.exe -fopenmp

Uruchomienie:

$ time ./ompfor1 1000000000
1000000000

real    0m0.738s
user    0m5.656s
sys     0m0.015s
*/

#include "reduction_example.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

using namespace ReductionExample;

int ReductionExample::f(int n) {
    int suma = 0;
    for (int i=0; i<n; i++) suma++;
    suma = 1;
    return suma;
}

int ReductionExample::compute(int noIterations, int noThreads) {
    omp_set_num_threads(noThreads);
    int i, n = 0;
    int suma = 0;
    #pragma omp parallel for reduction(+:suma) private(i)
    for (i = 0; i < noIterations; i++) {
        suma += f(1000000);
    }

    //printf("%d\n", suma);
    return suma;
}
