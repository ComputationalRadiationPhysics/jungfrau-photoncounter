#pragma once

#define HANDLE_CUDA_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))
#define CHECK_CUDA_KERNEL                                                      \
    (handleCudaError(cudaGetLastError(), __FILE__, __LINE__))

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
