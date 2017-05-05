#include "Kernel.hpp"

__global__ void calculate(uint32_t mapsize, uint64_t* pede, double* gain,
                          uint16_t* data, uint32_t num, uint16_t* photon)
{
    // locally save gain/ped values for the associated pixel
    uint16_t lPede[3];
    uint16_t lMovAvg;
    uint32_t lCounter;
    uint16_t lGain[3];

    // find id and copy gain/pede maps
    uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;

    lPede[0] = pede[id] & 0x000000000000ffff;
    lPede[1] = pede[mapsize + id] & 0x000000000000ffff;
    lPede[2] = pede[(2 * mapsize) + id] & 0x000000000000ffff;
    lMovAvg = (pede[id] & 0x00000000ffff0000) >> 16;
    lCounter = (pede[id] & 0xffffffff00000000) >> 32;
    lGain[0] = gain[id];
    lGain[1] = gain[mapsize + id];
    lGain[2] = gain[(mapsize * 2) + id];

    // calc the energy value for one pixel in each frame
    for (int i = 0; i < num; ++i) {
        // 8*(i++) is the header of each frame
        uint16_t dataword = data[(mapsize * i) + id + (8 * (i + 1))];
        uint16_t adc = dataword & 0x3fff;
        float energy;

        switch ((dataword & 0xc000) >> 14) {
        case 0:
            if (adc < 100) {
                // calibration for dark pixels
                lMovAvg = lMovAvg + adc - (lMovAvg / lCounter);
                if (lCounter < 4294000000)
                    lCounter++;

                lPede[0] = lMovAvg / lCounter;
            }
            else
                energy = (adc - lPede[0]) * lGain[0];
            break;
        case 1:
            energy = (lPede[1] - adc) * lGain[1];
            break;
        case 3:
            energy = (lPede[2] - adc) * lGain[2];
            break;
        default:
            energy = 0;
            break;
        }
        photon[(mapsize * i) + id + (8 * (i + 1))] = int((energy + 6.2) / 12.4);

        // copy the header
        if (threadIdx.x < 8) {
            photon[(mapsize * i) + (threadIdx.x * (i + 1))] =
                data[(mapsize * i) + (threadIdx.x * (i + 1))];
        }
    }

    // save new pedestal value
    pede[id] = ((uint64_t)lCounter << 32) | ((uint64_t)lMovAvg << 16) |
               (uint64_t)lPede[0];
}

__global__ void calibrate(uint32_t mapsize, uint32_t num, uint32_t currentnum,
                          uint16_t* data, uint64_t* pede)
{
    uint16_t id = blockIdx.x * blockDim.x + threadIdx.x;

    // 32 bit counter; 16 bit moving average; 16 bit offset
    // for calibration only average = offset
    uint32_t counter;
    uint16_t average;

    if (currentnum == 0) {
        counter = 0;
        average = 0;
    } else {
        counter = pede[id] & 0xffffffff00000000;
        average = pede[id] & 0x00000000ffff0000;
    }

    // base value for pedestal stage 0
    for (int i = currentnum; i < 1000 && i < num; i++) {
        average += data[(mapsize * i) + id] & 0x3fff;
        counter++;
    }

    average = round((double)average / counter);

    // combine all values into one 64 bit dataword, so we only need one map
    pede[id] = ((uint64_t)counter << 32) | ((uint64_t)average << 16) |
               (uint64_t)average;

    // base value for pedestal stage 1
    average = 0;
    counter = 0;
    for (int i = currentnum; i > 999 && i < 2000 && i < (999 + num); i++) {
        average += data[(mapsize * i) + id] & 0x3fff;
        counter++;
    }

    average = round((double)average / counter);

    pede[mapsize + id] = ((uint64_t)counter << 32) | ((uint64_t)average << 16) |
                         (uint64_t)average;

    // base value for pedestal stage 3
    average = 0;
    counter = 0;
    for (int i = currentnum; i > 1999 && i < 2999 + i < (1999 + num); i++) {
        average += data[(mapsize * i) + id] & 0x3fff;
        counter++;
    }

    average = round((double)average / counter);

    pede[(mapsize * 2) + id] = ((uint64_t)counter << 32) |
                               ((uint64_t)average << 16) | (uint64_t)average;
}
