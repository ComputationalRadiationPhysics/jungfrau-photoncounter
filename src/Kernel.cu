#include "Kernel.hpp"

__global__ void calculate(uint16_t* pede, double* gain, uint16_t* data, uint16_t num, float* energy) {
    const uint16_t arraysize = sizeof(pede) / sizeof(pede[0]);
    const uint16_t mapsize = arraysize / 3;
    __shared__ uint16_t sPede[arraysize];
    __shared__ uint16_t sGain[arraysize];

    uint16_t id = blockIdx.x * blockDim.x + threadIdx.x;

    sPede[id] = pede[id];
    sGain[id] = gain[id];

    __syncthreads();

    for(int i = 0; i < num; i++) {
        uint16_t dataword = data[(mapsize*i)+id];

        switch((dataword&0xc000) >> 14) {
            case 0: energy[(mapsize*i)+id] = 
                    (dataword&0x3fff - sPede[id]) * sGain[id];
                    break;
            case 1: energy[(mapsize*i)+id] =
                    (sPede[mapsize+id] - dataword&0x3fff) * 
                    sGain[mapsize*id];
                    break;
            case 3: energy[(mapsize*i)+id] =
                    (sPede[(2*mapsize)+id] - dataword&0x3fff) *
                    sGain[(2*mapsize)+id];
                    break;
            default: 
                    energy[(mapsize*i)+id] = 0;
                    break;
        }
    } 
}

void calibrate(uint16_t* data, uint16_t num, uint16_t* pede) {
    /*TODO
    const uint16_t arraysize = sizeof(pede) / 16;
    __shared__ uint16_t map[arraysize];

    uint16_t id = blockIdx.x * blockDim.x + threadIdx.x;
    
    map[0] = 0;
    pede[0] = 0;
    for(int i = 1; i < num; i++) {
        pede[id] = (map[id] + data[i]&0x3fff - (map[id-1] / i)) / i;
    }
    */
}

//delete this, it's just so it compiles with nvcc
int main() {
    return 0;
}
