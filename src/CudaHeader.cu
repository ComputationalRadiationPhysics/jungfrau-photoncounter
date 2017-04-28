#include "CudaHeader.hpp"

void handleCudaError(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess) {
        char errorString[1000];
        snprintf(errorString, 1000,
                 "FATAL ERROR (CUDA, %d): %s in %s at line %d!\n", error,
                 cudaGetErrorString(error), file, line);
        fputs(errorString, stderr);
        exit(EXIT_FAILURE);
    }
}
