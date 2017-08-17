#pragma once

#include "../Config.hpp"

__global__ void calculate(uint16_t* data, pedestal* pede, double* gain,
                            uint32_t num, uint16_t* photon);
