#pragma once

#include <cstdio>

#define HANDLE_CUDA_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))
#define CHECK_CUDA_KERNEL                                                      \
    (handleCudaError(cudaGetLastError(), __FILE__, __LINE__))

void handleCudaError(cudaError_t error, const char* file, int line);
