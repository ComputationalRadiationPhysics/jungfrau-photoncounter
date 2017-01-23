#include "Kernel.hpp"

__global__ void calculate(uint16_t mapsize, uint16_t* pede, double* gain,
                          uint16_t* data, uint16_t num, uint16_t* photon)
{
    extern __shared__ uint16_t sPede[];
    extern __shared__ double sGain[];

    uint16_t id = blockIdx.x * blockDim.x + threadIdx.x;

    sPede[threadIdx.x] = pede[threadIdx.x];
    sPede[blockDim.x + threadIdx.x] = pede[blockDim.x + threadIdx.x];
    sPede[(blockDim.x * 2) + threadIdx.x] = pede[(blockDim.x * 2) + threadIdx.x];
    sGain[threadIdx.x] = gain[threadIdx.x];
    sGain[blockDim.x + threadIdx.x] = gain[blockDim.x + threadIdx.x];
    sGain[(blockDim.x * 2) + threadIdx.x] = gain[(blockDim.x * 2) + threadIdx.x];

    __syncthreads();
    

    for (int i = 0; i < num; i++) {
        uint16_t dataword = data[(mapsize * i) + id];
        float energy;

        switch ((dataword & 0xc000) >> 14) {
        case 0:
            energy =
                (dataword & 0x3fff - sPede[threadIdx.x]) * sGain[threadIdx.x];
            break;
        case 1:
            energy =
                (sPede[blockDim.x + threadIdx.x] - dataword & 0x3fff) * 
                sGain[blockDim.x + threadIdx.x];
            break;
        case 3:
            energy =
                (sPede[(2 * blockDim.x) + threadIdx.x] - dataword & 0x3fff) *
                sGain[(2 * blockDim.x) + threadIdx.x];
            break;
        default:
            energy = 0;
            break;
        }

        photon[(mapsize * i) + id] = int((energy + 6.2) / 12.4);
    }
}

__global__ void calibrate(uint16_t mapsize, uint16_t* data, uint16_t num,
                          uint16_t* pede)
{
    extern __shared__ uint16_t sPede[];

    uint16_t id = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < 1000; i++) {
        if (i == 0) {
            sPede[threadIdx.x] = data[(mapsize * i) + id] & 0x3fff;
        }
        else {
            sPede[threadIdx.x] += data[(mapsize * i) + id] & 0x3fff;
        }
    }

    for (int i = 1000; i < num; i++) {
        sPede[threadIdx.x] =
            (sPede[threadIdx.x] + data[(mapsize * i) + id] & 0x3fff) - 
            (sPede[threadIdx.x] / i);
    }

    pede[id] = round((double)sPede[threadIdx.x] / 1000);
}
