#pragma once

#include "../Config.hpp"

__global__ void sum(uint16_t* data, uint16_t amount, uint32_t num,
                    uint64_t* sum);
