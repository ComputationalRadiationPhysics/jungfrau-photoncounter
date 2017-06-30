#include "CalibrationKernel.hpp"
#include "Settings.hpp"

__global__ void calibrate(uint16_t* data, uint64_t* pede, uint32_t mapsize,
                          uint32_t currentnum)
{
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    uint16_t stage = (uint16_t)currentnum / framesPerStage;

    size_t counter = currentnum;
    while (counter < (stage * framesPerStage)) {
        pede[(stage * mapsize) + id].movAvg +=
            data[(mapsize * (i - currentnum)) + id + (frameoffset * (i - 1))] &
            0x3fff;
        counter++;
    }
    if (counter == ((stage + 1) * framesPerStage)) {
        pede[(stage * mapsize) + id].counter = framesPerStage;
        pede[(stage * mapsize) + id].movAvg /= framesPerStage;
        pede[(stage * mapsize) + id].value =
            pede[(stage * mapsize) + id].movAvg;
    }
}
