#pragma once

#include <cstdint>
#include <limits>

const std::size_t GPU_FRAMES = 1000;
const std::size_t SUM_FRAMES = 100;
const std::size_t STREAMS_PER_GPU = 2;
const std::size_t FRAME_HEADER_SIZE = 16;
const std::size_t DIMX = 1024;
const std::size_t DIMY = 512;
const std::size_t MAPSIZE = DIMX * DIMY;
const std::size_t FRAMESPERSTAGE = 1000;
const std::size_t FRAMEOFFSET = FRAME_HEADER_SIZE / 2;
const float BEAMCONST = 6.2;
const float PHOTONCONST = (1. / 12.4);
const std::size_t MAXINT = std::numeric_limits<uint32_t>::max();

struct pedestal{
    uint32_t counter;
    uint16_t value;
    uint32_t movAvg;

    pedestal() : counter(0), value(0), movAvg(0) {}
};

using DataType = uint16_t;
using GainType = double;
using PedestalType = pedestal;
using PhotonType = uint16_t;
using PhotonSumType = uint64_t;

// TODO: remove this debugging statement later
#include <iostream>
#define DEBUG(msg)                                                             \
    (std::cout << __FILE__ << "[" << __LINE__ << "]:\t" << msg << std::endl)

