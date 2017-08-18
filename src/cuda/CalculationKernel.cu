#include "CalculationKernel.hpp"

__global__ void calculate(uint16_t* data, pedestal* pede, double* gainmap,
                            uint32_t num, uint16_t* photon)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
    
    uint16_t pedestal[3];
    uint32_t pCounter;
    uint32_t pMovAvg;

    double gain[3];

    for(int i = 0; i < 3; i++) {
        pedestal[i] = pede[(i * MAPSIZE) + id].value;
        gain[i] = gainmap[(i * MAPSIZE) + id];
    }
    pCounter = pede[id].counter;
    pMovAvg = pede[id].movAvg;

    uint16_t dataword;
    uint16_t adc;
    float energy;

    for (int i = 0; i < num; ++i) {
        dataword = data[(MAPSIZE * i) + id + (FRAMEOFFSET * (i + 1))];
        adc = dataword & 0x3fff;

        switch ((dataword & 0xc000) >> 14) {
        case 0:
            if (adc < 100) {
                // calibration for dark pixels
                pMovAvg = pMovAvg + adc - (pMovAvg / pCounter);
                if (pCounter < MAXINT)
                    pCounter++;

                pedestal[0] = pMovAvg / pCounter;
            }
            energy = (adc - pedestal[0]) / gain[0];
            if (energy < 0) energy = 0;
            break;
        case 1:
            energy = (-1) * (pedestal[1] - adc) / gain[1];
            if (energy < 0) energy = 0;
            break;
        case 3:
            energy = (-1) * (pedestal[2] - adc) / gain[2];
            if (energy < 0) energy = 0;
            break;
        default:
            energy = 0;
            break;
        }
        photon[(MAPSIZE * i) + id + (FRAMEOFFSET * (i + 1))] = 
            int((energy + BEAMCONST) * PHOTONCONST);
        
        // copy the header
        if (threadIdx.x < 8) {
        photon[(MAPSIZE * i) + (threadIdx.x * (i + 1))] =
        data[(MAPSIZE * i) + (threadIdx.x * (i + 1))];
        }
    }
    // save new pedestal value
    pede[id].value = pedestal[0];
    pede[id].counter = pCounter;
    pede[id].movAvg = pMovAvg;
}
