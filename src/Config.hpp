#pragma once

#include <cstdint>

// TODO: test different sizes
const std::size_t GPU_FRAMES = 1000;
const std::size_t SUM_FRAMES = 100;
const std::size_t STREAMS_PER_GPU = 2;
const std::size_t FRAME_HEADER_SIZE = 16;
const std::size_t DIMX = 1024;
const std::size_t DIMY = 512;

using DataType = uint16_t;
using GainType = double;
using PedestalType = uint64_t;
using PhotonType = uint16_t;
using PhotonSumType = uint64_t;

// TODO: remove this debugging statement later
#include <iostream>
#define DEBUG(msg)                                                             \
    (std::cout << __FILE__ << "[" << __LINE__ << "]:\t" << msg << std::endl)

