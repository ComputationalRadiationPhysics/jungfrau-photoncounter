#pragma once

__global__ void calculate(uint16_t* data, uint64_t* pede, double* gain,
                            uint32_t num, uint16_t* photon);
