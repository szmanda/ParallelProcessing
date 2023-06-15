#include "kernel.cuh"

__host__ __device__ char* toChar(const std::vector<char>& olig, int l) {
    char* result = new char[l + 1];
    for (int i = 0; i < l; i++)
        result[i] = olig[i];
    result[l] = '\0';
    return result;
}
__device__ char* toCharDevice(const char* olig, int l) {
    char* result = new char[l + 1];
    for (int i = 0; i < l; i++)
        result[i] = olig[i];
    result[l] = '\0';
    return result;
}

__host__ void printSolution(int* solution, int** offsets, const std::vector<std::vector<char>>& oligs, int length, int start) {
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
__device__ void printSolution(int* solution, int* offsets_flat, int s, char* oligs_flat, int length, int start) {
    length--; // without last vertex;
    int l = 10;
    printf("\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\033[15A");
    for (int k = 0; k + 2 < start % 15; k++) printf("\n");
    for (int j = start; j < length; j++) {
        for (int r = 0; r < l; r++) printf("%c", oligs_flat[solution[j] * l + r]);
        printf("\033[10D\033[B");
        for (int k = 0; k < offsets_flat[solution[j] * s + solution[j + 1]]; k++)
            printf("\033[C");
        if (j > 0 && j % 15 == 0) printf("\033[15A");
        if (j + 1 == length) for (int k = length % 15; k < 15; k++) printf("\n");
    }
    printf("\n\n");
}

__global__ void kernelTabuSearch(
    int* solution_shared,
    const int* offsets,
    const int s,
    const int n,
    const char* oligs_flat,
    int* tabuFragments_shared, // [tabuLimit * TabuFragmentLength]
    const int tabuLimit, // number of fragments in tabu list
    const int tabuFragmentLength, // length of tabu fragments
    int* tabuCount,
    int* tabuId,
    bool* used_shared
) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
    int* solution = &solution_shared[thread * s];
    int* tabuFragments = &tabuFragments_shared[thread * tabuLimit * tabuFragmentLength];
    bool* used = &used_shared[thread * s];

    /// Simple Tabu Search from one of the vertices
    // printf("== BEGIN KernelTabuSearch %d == \t start: %d %.10s, s: %d, n: %d, tabuLimit: %d, tabuFragmentLength: %d, tabuCount: %d, tabuId %d,\r\n",
    //    thread, solution[0], &oligs_flat[solution[0] * 10], s, n, tabuLimit, tabuFragmentLength, *tabuCount, *tabuId);
    
    // populating tabu list (for the first iteration)
    if (*tabuCount == 0) {
        // printf("populating tabu list\r\n");
        for (int j = 0; j < s / tabuFragmentLength; j++) {
            for (int k = 0; k < tabuFragmentLength; k++) {
                tabuFragments[j * tabuFragmentLength + k] = solution[tabuFragmentLength * j + k];
            }
            *tabuCount += 1;
            *tabuId += 1;
        }
    }
    

    // modifying previus solution considering tabu list
    // this is for debug purpouses, the actual algorythm works on 'solution' in place
    int* prevSolution = new int[s];
    for (int i = 0; i < s; i++) {
        prevSolution[i] = solution[i];
    }
    int infrCount = 0;
    const int infrLimit = 10;
    int infringementFragments[infrLimit];
    int infringementId[infrLimit];
    used[solution[0]] = true;
    for (int j = 1; j < s; j++) {
        const int bestLimit = 10;
        int best[bestLimit];
        int bestCount = 0;
        for (int k = 0; k < s; k++) {
            if (used[k]) continue;
            int id = bestCount;
            int b = bestCount - 1;
            while (b >= 0 && offsets[solution[j - 1] * s + k] < offsets[solution[j - 1] * s + best[b]])
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
        int seed = (int)((float)j / 17 * 1000) + solution[(((*tabuId + int(logf(j + 104) * 1000))) % (*tabuCount + 1)) * (j % tabuFragmentLength + 1)];
        // printf("seed: %d <-- %d, %d\n", seed, (int)((float)j/17*1000), solution[(((*tabuId + int(logf(j+104)*1000))) % (*tabuCount+1)) * (j % tabuFragmentLength + 1)]);
        // if (seed % 10 == 0) printf("X"); else printf("_"); // does look pretty randomized
        bool chosen = false;
        for (int k = 0; k < bestCount; k++) {
            int probModulo = 0;
            for (int kk = 0; kk < infrCount; kk++) {
                infringementId[kk] += (infringementId[kk] < tabuFragmentLength - 1) ? 1 : 0;
                if (tabuFragments[infringementFragments[kk] * tabuFragmentLength + infringementId[kk]] == best[k]) {
                    probModulo = tabuFragmentLength - infringementId[kk];
                    if (probModulo > 0 && seed % probModulo == 0) {
                        solution[j] = best[k];
                        used[best[k]] = true;
                        break;
                    }
                }
            }
            if (probModulo == 0 && infrCount > 0) {
                infrCount = 0;
                if (seed % tabuFragmentLength == 0) { // probability of ignoring the first infringement
                    solution[j] = best[k];
                    used[best[k]] = true;
                }
            }
            // adding new infringements
            for (int kk = 0; kk < *tabuCount; kk++)
                if (tabuFragments[kk * tabuFragmentLength + 0] == best[k]) {
                    if (used[best[k]] && infrCount < infrLimit) {
                        infringementFragments[infrCount] = k;
                        infringementId[infrCount] = 0;
                        infrCount++;
                    }
                }
            if (infrCount == 0) {
                solution[j] = best[k];
                used[best[k]] = true;
            }
            if (used[best[k]]) {
                //printf("%d", k);
                chosen = true;
                break;
            }
        }
        if (!chosen) {
            //printf("Fallback greedy\n");
            //printf("_");
            solution[j] = best[0];
            used[best[0]] = true;
        }


        //if (used[solution[j]]) {
        //    /*printf("used: %d, prev: %d", solution[j], prevSolution[j]);
        //    if (chosen) printf("<chosen>");
        //    printf("\n");*/
        //}
        //else {
        //    printf("ERR, solution: %d, prev: %d, best: [", solution[j], prevSolution[j]);
        //    for (int rr = 0; rr < bestCount; rr++) printf("%d, ", best[rr]); printf("], ");
        //    if (chosen) printf("<chosen>");
        //    printf("\n");
        //}
        delete[] best;
    }
    // updating the tabu list
    // printf("TabuCount: %d, now adding new entries:\n", *tabuCount);
    for (int j = 0; j < s / tabuFragmentLength; j++) {
        // printf("_%d", *tabuCount);
        for (int k = 0; k < tabuFragmentLength; k++) {
            tabuFragments[(*tabuId++ % *tabuCount) * tabuFragmentLength + k] = solution[tabuFragmentLength * j + k];
        }
        if (*tabuCount < tabuLimit) *tabuCount += 1;
    }
    // evaluating the solution (first n oligs)
    int cost = 0;
    int nn = n < s ? n : s;
    for (int j = 1; j < nn; j++) { cost += offsets[(j - 1) * s + j]; }
    int changed_from_prev = 0;
    for (int j = 1; j < nn; j++) {
        if (solution[j] != prevSolution[j]) {
            changed_from_prev++;
            /*printf("\n%d: \t %d --> %d\t\t", j, prevSolution[j], solution[j]);
            for (int i = 0; i < 10; i++) printf("%c", oligs_flat[prevSolution[j] * 10 + i]);
            printf(" --> ");
            for (int i = 0; i < 10; i++) printf("%c", oligs_flat[solution[j] * 10 + i]);
            if (changed_from_prev > 30) break;*/
        }
    }
    /*int unused_check = 0;
    for (int i = 0; i < s; i++) if (used[i] == false) unused_check++;
    if (unused_check > -1) printf("There are %d unused oligs!!!\n", unused_check);*/

    // printf("Cost of new solution: %d, oligs that are on different positions: %d\n", cost, changed_from_prev);

    //printf("%s\n", oligs_flat);

    // delete[] prevSolution;
    delete[] infringementFragments;
    delete[] infringementId;
    //delete[] used;

    // printf("== END KernelTabuSearch %d ==\n", thread);
}