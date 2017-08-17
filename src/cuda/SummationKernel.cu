#include "SummationKernel.hpp"

__global__ void sum(uint16_t* data, uint16_t amount, uint32_t num,
                    uint64_t* sum)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t summation = 0;

    for (int i = 0; i < num; i++) {
        summation += data[(i * MAPSIZE) + id];
        if (i % amount) {
            sum[((i / amount) * MAPSIZE) + id] = summation;
            summation = 0;
        }
    }
}
