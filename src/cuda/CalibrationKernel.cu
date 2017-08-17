#include "CalibrationKernel.hpp"
#include <iostream>

__global__ void calibrate(uint16_t* data, pedestal* pede, uint32_t currentnum)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    size_t counter = currentnum;

    while (counter - currentnum < GPU_FRAMES) {
        uint16_t stage = counter / FRAMESPERSTAGE;

        while (counter < ((stage + 1) * FRAMESPERSTAGE) &&
               counter < ((stage * FRAMESPERSTAGE) + GPU_FRAMES)) {
            pede[(stage * MAPSIZE) + id].movAvg +=
                data[(MAPSIZE * (counter - currentnum)) + id +
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
