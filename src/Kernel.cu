#include "Kernel.hpp"

#include <cstdio>

__global__ void calculate(uint16_t mapsize, uint16_t* pede, double* gain,
                          uint16_t* data, uint16_t num, uint16_t* photon)
{
    extern __shared__ uint16_t sPede[];
    extern __shared__ double sGain[];

    uint16_t id = blockIdx.x * blockDim.x + threadIdx.x;
	/*
    sPede[threadIdx.x] = pede[id];
    sPede[mapsize + id] = pede[mapsize + id];
    sPede[mapsize * 2 + id] = pede[mapsize * 2 + id];
    sGain[id] = gain[id];
    sGain[mapsize + id] = gain[mapsize + id];
    sGain[mapsize * 2 + id] = gain[mapsize * 2 + id];

    __syncthreads();*/

    for (int i = 0; i < num; i++) {
        uint16_t dataword = data[(mapsize * i) + id];
		uint16_t adc = dataword & 0x3fff;
        float energy;

        switch ((dataword & 0xc000) >> 14) {
        case 0:
			//TODO: use shared memory here
            energy = (adc - pede[id]) * gain[id];
            break;
        case 1:
            energy = (pede[mapsize + id] - adc) * gain[mapsize + id];
            break;
        case 3:
            energy = (pede[mapsize * 2 + id] - adc) * gain[mapsize * 2 + id];
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
