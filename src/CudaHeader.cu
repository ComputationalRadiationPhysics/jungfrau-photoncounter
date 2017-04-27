#include "CudaHeader.hpp"
#include <stdio.h>

void handleCudaError(cudaError_t error, const char* file, int line) {
    if(error == cudaSuccess) return;
    printf("<%s>:%i",file,line);
    printf(" %s\n", cudaGetErrorString(error));
}
