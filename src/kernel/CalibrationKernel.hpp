#pragma once

__global__ void calibrate(uint16_t* data, uint64_t* pede, uint32_t mapsize,
                            uint32_t currentnum);
