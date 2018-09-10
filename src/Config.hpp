#pragma once

#include <alpaka/alpaka.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

// general settings
constexpr std::size_t FRAMESPERSTAGE_G0 = 1000;
constexpr std::size_t FRAMESPERSTAGE_G1 = 1000;
constexpr std::size_t FRAMESPERSTAGE_G2 = 999;

constexpr std::size_t FRAME_HEADER_SIZE = 16;
constexpr std::size_t FRAMEOFFSET = FRAME_HEADER_SIZE / 2;
constexpr std::size_t DIMX = 1024;
constexpr std::size_t DIMY = 512;
constexpr std::size_t MAPSIZE = DIMX * DIMY;
constexpr std::size_t SINGLEMAP = 1;
constexpr std::size_t SUM_FRAMES = 100;
constexpr std::size_t DEV_FRAMES = 1000;
constexpr std::size_t PEDEMAPS = 3;
constexpr std::size_t GAINMAPS = 3;
constexpr float BEAMCONST = 6.2;
constexpr float PHOTONCONST = (1. / 12.4);
constexpr std::size_t MAXINT = std::numeric_limits<uint32_t>::max();

constexpr int CLUSTER_SIZE = 3;

// data types
/*template <typename TData, typename TAlpaka, typename TDim, typename TSize>
struct Maps {
    std::size_t numMaps;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, TData, TDim, TSize> data;
    Maps()
        : numMaps(0),
          data(alpaka::mem::buf::alloc<TData, typename TSize>(
              alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
              0lu)){};
};
*/
struct FrameHeader {
    std::uint64_t frameNumber;
    std::uint64_t bunchId;
};

template <typename TData> struct Frame {
    FrameHeader header;
    TData data[DIMX * DIMY];
};

struct Pedestal {
    std::size_t counter;
    double mean;
    double M2;
    double stddev;
};

struct Cluster {
    std::uint64_t frameNumber;
    std::int16_t x;
    std::int16_t y;
    std::int32_t data[CLUSTER_SIZE * CLUSTER_SIZE];
};

struct ClusterArray {
    std::size_t used;
    Cluster* clusters;
};

template <typename T, typename TAlpaka, typename TDim, typename TSize> struct FramePakage {
    std::size_t numFrames;
    alpaka::mem::buf::
        Buf<typename TAlpaka::DevHost, T, TDim, TSize>
            data;

    FramePakage()
        : numFrames(0),
          data(alpaka::mem::buf::alloc<T, TSize>(
              alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
              static_cast<TSize>(1u)))
    {
    }
};

using DetectorData = Frame<std::uint16_t>;
using PhotonMap = DetectorData;
using PhotonSumMap = Frame<std::uint64_t>;
using DriftMap = Frame<std::uint32_t>;
using GainStageMap = Frame<char>;
using MaskMap = Frame<bool>;
using EnergyMap = Frame<float>;
using GainMap = float[DIMX * DIMY];
using PedestalMap = Pedestal[DIMX * DIMY];

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
    img.open(path + ".txt");
    for (std::size_t j = 0; j < 512; j++) {
        for (std::size_t k = 0; k < 1024; k++) {
            int h = int(data[frame_number].data[(j * 1024) + k] * 10);
            img << h << " ";
        }
        img << "\n";
    }
    img.close();
#endif
}
