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
    int* solution,
    int *offsets,
    const int s,
    char* oligs_flat,
    int* tabuFragments, // [tabuLimit * TabuFragmentLength]
    int tabuLimit, // number of fragments in tabu list
    int tabuFragmentLength, // length of tabu fragments
    int* tabuCount,
    int* tabuId
) {
    /// Simple Tabu Search from one of the vertices
    printf("+----- KernelTabuSearch -----+\r\n\n");
    printf("s: %d, tabuLimit: %d, tabuFragmentLength: %d, tabuCount: %d, tabuId %d,\r\n",
        s, tabuLimit, tabuFragmentLength, *tabuCount, *tabuId);
    // populating tabu list (for the first iteration)
    if (*tabuCount == 0) {
        printf("populating tabu list\r\n");
        for (int j = 0; j < s / tabuFragmentLength; j++) {
            for (int k = 0; k < tabuFragmentLength; k++) {
                tabuFragments[j * tabuFragmentLength + k] = solution[tabuFragmentLength * j + k];
            }
            *tabuCount += 1;
            *tabuId += 1;
        }
    }
    
    printf("\n");


    // modifying previus solution considering tabu list
    // this is for debug purpouses, the actual algorythm works on 'solution' in place
    int* prevSolution = new int[s];
    for (int i = 0; i < s; i++) {
        prevSolution[i] = solution[i];
    }
    int infrCount = 0;
    int infrLimit = 0;
    int* infringementFragments = new int[infrLimit];
    int* infringementId = new int[infrLimit];
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
            while (b >= 0 && offsets[solution[j - 1] * s + k] < offsets[solution[j - 1] * s + best[b]])
                id = b--;
            if (id < bestCount || bestCount < bestLimit) {
                for (int ll = bestCount - 1; ll > id; ll--)
                    best[ll] = best[ll - 1];
                best[id] = k;
                if (bestCount < bestLimit) bestCount++;
            }
        }
        if (bestCount == 0) {
            printf("ERR: did not find any best solutions!!! that's dangerous");
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
                    if (seed % probModulo == 0) {
                        solution[j] = best[k];
                        used[best[k]] = true;
                    }
                    break;
                }
            }
            if (probModulo == 0 && infrCount > 0) {
                infrCount = 0;
            }

            if (seed % tabuFragmentLength == 0) {
                solution[j] = best[k];
                used[best[k]] = true;
            }
            // adding new infringements
            bool infringing = false;
            for (int kk = 0; kk < *tabuCount; kk++)
                if (tabuFragments[kk * tabuFragmentLength + 0] == best[k]) {
                    if (used[best[k]] && infrCount < infrLimit) {
                        infringementFragments[infrCount] = k;
                        infringementId[infrCount] = 0;
                        infrCount++;
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
            //printf("Fallback greedy\n");
            printf("_");
            solution[j] = best[0];
            used[best[0]] = true;
        }
    }
    // updating the tabu list
    printf("\nTabuCount: %d, now adding new entries:\n", *tabuCount);
    for (int j = 0; j < s / tabuFragmentLength; j++) {
        printf("_%d", *tabuCount);
        for (int k = 0; k < tabuFragmentLength; k++) {
            tabuFragments[(*tabuId++ % *tabuCount) * tabuFragmentLength + k] = solution[tabuFragmentLength * j + k];
        }
        if (*tabuCount < tabuLimit) *tabuCount += 1;
    }
    // evaluating the solution
    int cost = 0;
    for (int j = 1; j < s; j++) { cost += offsets[(j - 1) * s + j]; }
    int changed_from_prev = 0;
    for (int j = 1; j < s; j++) {
        if (solution[j] != prevSolution[j]) {
            changed_from_prev++;
            printf("\n%d: \t %d --> %d\t\t", j, prevSolution[j], solution[j]);
            for (int i = 0; i < 10; i++) printf("%c", oligs_flat[prevSolution[j] * 10 + i]);
            printf(" --> ");
            for (int i = 0; i < 10; i++) printf("%c", oligs_flat[solution[j] * 10 + i]);
            if (changed_from_prev > 30) break;
        }
    }

    printf("\n\nCost of new solution: %d, oligs that are on different positions: %d", cost, changed_from_prev);

    /*printf("\nSolution for ");
    for (int i = 0; i < 10; i++) printf("%c", oligs_flat[i]);
    printf("\n\n");
    printSolution(solution, offsets, s, oligs_flat, 100);
    printf("\t\t\t . . .\r\n\n\n");
    printSolution(solution, offsets, s, oligs_flat, s / 2 + 80, s / 2);
    printf("\t\t\t . . .\r\n\n\n");
    printSolution(solution, offsets, s, oligs_flat, s, s - 30);
    printf("\r\n\n\n");*/

    //printf("%s\n", oligs_flat);

    printf("\n\n+----- Exiting KernelTabuSearch ------+\n\n");
}