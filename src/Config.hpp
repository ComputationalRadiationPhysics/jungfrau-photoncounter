#pragma once

#include <alpaka/alpaka.hpp>
#include <chrono>
#include <iostream>
#include <fstream>

// general settings
const std::size_t FRAMESPERSTAGE_G0 = 1000;
const std::size_t FRAMESPERSTAGE_G1 = 1000;
const std::size_t FRAMESPERSTAGE_G2 = 999;

const std::size_t FRAME_HEADER_SIZE = 16;
const std::size_t FRAMEOFFSET = FRAME_HEADER_SIZE / 2;
const std::size_t DIMX = 1024;
const std::size_t DIMY = 512;
const std::size_t MAPSIZE = DIMX * DIMY;
const std::size_t SUM_FRAMES = 100;
const std::size_t DEV_FRAMES = 1000;
const std::size_t PEDEMAPS = 3;
const std::size_t GAINMAPS = 3;
const float BEAMCONST = 6.2;
const float PHOTONCONST = (1. / 12.4);
const std::size_t MAXINT = std::numeric_limits<uint32_t>::max();

// data types
template <typename TData, typename TAlpaka> struct Maps {
    std::size_t numMaps;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          TData,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        data;
    bool header;
    Maps()
        : numMaps(0),
          data(alpaka::mem::buf::alloc<TData, typename TAlpaka::Size>(
              alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
              0lu)),
          header(false){};
};

<<<<<<< Updated upstream
template <typename TData> struct Frame {
    uint64_t framenumber;
    uint64_t bunchid;
    TData imagedata[DIMX * DIMY];
}; 
using GainStage = Frame<std::uint8_t>;
using Drift = Frame<std::uint32_t>;
using Data = Frame<std::uint16_t>;
using Charge = Frame<double>;
using Mask = bool[DIMX * DIMY]; 
using Gain = double[DIMX * DIMY];
using Photon = Frame<std::uint16_t>;
using PhotonSum = Frame<std::uint64_t>;

struct PedestalStruct {
    std::size_t counter;
    double mean;
    double M2;
    double stddev;
};

using Pedestal = PedestalStruct[DIMX * DIMY];

// debug statements
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds ms;
static Clock::time_point t;

#define SHOW_DEBUG true

#if (SHOW_DEBUG)
#include <iostream>
#define DEBUG(msg)                                                             \
    (std::cout << __FILE__ << "[" << __LINE__ << "]:\n\t"                      \
               << (std::chrono::duration_cast<ms>((Clock::now() - t))).count() \
               << " ms\n\t" << msg << std::endl)
#else
#define DEBUG(msg)
#endif

#if (SHOW_DEBUG)
#include "Bitmap.hpp"
#endif
template <typename TBuffer>
void save_image(std::string path, TBuffer* data, std::size_t frame_number)
{
#if (SHOW_DEBUG)
    std::ofstream img;
    img.open (path + ".txt");
    for (std::size_t j = 0; j < 512; j++) {
        for (std::size_t k = 0; k < 1024; k++) {
            int h = int(data[frame_number].imagedata[(j * 1024) + k] *10);
            img << h << " ";
        }
    img << "\n";
    }
    img.close();
#endif
}
