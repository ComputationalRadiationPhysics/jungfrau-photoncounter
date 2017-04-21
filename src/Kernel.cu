#include "Kernel.hpp"

#include <cstdio>

__global__ void calculate(uint32_t mapsize, uint16_t* pede, double* gain,
                          uint16_t* data, uint32_t num, uint16_t* photon)
{
    extern __shared__ double shared[];
	double* sPede = &shared[0];
	double* sGain = &shared[blockDim.x * 3];
 
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	
    sPede[threadIdx.x] = pede[id];
    sPede[blockDim.x + threadIdx.x] = pede[mapsize + id];
    sPede[(blockDim.x * 2) + threadIdx.x] = pede[(mapsize * 2) +id];
    sGain[threadIdx.x] = gain[id];
    sGain[blockDim.x + threadIdx.x] = gain[mapsize + id];
    sGain[(blockDim.x * 2) + threadIdx.x] = gain[(mapsize * 2) + id];

    __syncthreads();
		
    for (int i = 0; i < num; ++i) {
        uint16_t dataword = data[(mapsize * i) + id + (8 * (i+1))];
		uint16_t adc = dataword & 0x3fff;
        float energy;

        switch ((dataword & 0xc000) >> 14) {
		case 0:
            energy =
                (adc - sPede[threadIdx.x]) * sGain[threadIdx.x];
            break;
        case 1:
            energy =
                (sPede[blockDim.x + threadIdx.x] - adc) * 
                sGain[blockDim.x + threadIdx.x];
            break;
        case 3:
            energy =
                (sPede[(2 * blockDim.x) + threadIdx.x] - adc) *
                sGain[(2 * blockDim.x) + threadIdx.x];
				break;
        default:
            energy = 0;
            break;
        }
        photon[(mapsize * i) + id + (8 * (i+1))] = int((energy + 6.2) / 12.4);
		 
        if(threadIdx.x < 8) {
            photon[(mapsize * i) + id + (threadIdx.x * (i+1))] = 
                data[(mapsize * i) + id + (threadIdx.x * (i+1))];
		}
	}
}

__global__ void calibrate(uint32_t mapsize, uint16_t* data, uint32_t num,
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
