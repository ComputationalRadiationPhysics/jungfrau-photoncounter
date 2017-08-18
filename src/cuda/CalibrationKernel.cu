#include "CalibrationKernel.hpp"
#include <iostream>

__global__ void calibrate(uint16_t* data, pedestal* pede, uint32_t numframes)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    size_t counter = 0;
    uint16_t stage = 0;

    while (counter < numframes) {
        for (size_t i = 0; i < 3; i++) {
            if (pede[(i * MAPSIZE) + id].counter == FRAMESPERSTAGE) stage++;
        }

        while (counter < ((stage + 1) * FRAMESPERSTAGE) &&
               counter < ((stage * FRAMESPERSTAGE) + numframes)) {
            pede[(stage * MAPSIZE) + id].movAvg +=
                data[(MAPSIZE * (counter)) + id +
                     (FRAMEOFFSET * (counter + 1))] &
                0x3fff;
            pede[(stage * MAPSIZE) + id].counter ++;
            counter++;
        }
            if (pede[(stage * MAPSIZE) + id].counter == FRAMESPERSTAGE)
            {
                pede[(stage * MAPSIZE) + id].movAvg /= FRAMESPERSTAGE;
                pede[(stage * MAPSIZE) + id].value =
                    pede[(stage * MAPSIZE) + id].movAvg;
            }
    }
}
