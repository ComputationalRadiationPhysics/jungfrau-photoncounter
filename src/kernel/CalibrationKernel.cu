#include "CalibrationKernel.hpp"
#include "Settings.hpp"

__global__ void calibrate(uint16_t* data, uint64_t* pede, uint32_t currentnum)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    uint16_t stage = (uint16_t)currentnum / FRAMESPERSTAGE;

    size_t counter = currentnum;
    while (counter < (stage * framesPerStage)) {
        pede[(stage * MAPSIZE) + id].movAvg +=
            data[(MAPSZIE * (i - currentnum)) + id + (FRAMEOFFSET * (i - 1))] &
            0x3fff;
        counter++;
    }
    if (counter == ((stage + 1) * framesPerStage)) {
        pede[(stage * MAPSIZE) + id].counter = framesPerStage;
        pede[(stage * MAPSIZE) + id].movAvg /= framesPerStage;
        pede[(stage * MAPSIZE) + id].value =
            pede[(stage * MAPSIZE) + id].movAvg;
    }
}
