#pragma once

#include "../Config.hpp"

__global__ void calibrate(uint16_t* data, pedestal* pede, uint32_t currentnum);
