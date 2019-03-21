#pragma once

#include <alpaka/alpaka.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

#include "CheapArray.hpp"

// general settings
constexpr std::size_t FRAMESPERSTAGE_G0 = 1000;
constexpr std::size_t FRAMESPERSTAGE_G1 = 0;
constexpr std::size_t FRAMESPERSTAGE_G2 = 0;

constexpr std::size_t FRAME_HEADER_SIZE = 16;
constexpr std::size_t DIMX = 400;
constexpr std::size_t DIMY = 400;
constexpr std::size_t SUM_FRAMES = 10;
constexpr std::size_t DEV_FRAMES = 300;
constexpr std::size_t PEDEMAPS = 3;
constexpr std::size_t MOVING_STAT_WINDOW_SIZE = 100;
constexpr std::size_t GAINMAPS = 3;
constexpr float BEAMCONST = 6.2;
constexpr float PHOTONCONST = (1. / 12.4);
constexpr int CLUSTER_SIZE = 3;
constexpr int C = 5;

// derived settings
constexpr std::size_t MAPSIZE = DIMX * DIMY;
constexpr std::size_t SINGLEMAP = 1;
constexpr std::size_t MAXINT = std::numeric_limits<uint32_t>::max();
constexpr char MASKED_VALUE = 4;

// maximal number of clusters possible:
constexpr uint64_t MAX_CLUSTER_NUM = (DIMX - CLUSTER_SIZE + 1) *
                                     (DIMY - CLUSTER_SIZE + 1) /
                                     ((CLUSTER_SIZE / 2) * (CLUSTER_SIZE / 2));

static_assert(
    FRAMESPERSTAGE_G0 >= MOVING_STAT_WINDOW_SIZE,
    "Moving stat window size is bigger than the frames supplied for the "
    "callibration of the pedestal values for the first gain stage. ");

// maximal number of seperated clusters:
constexpr uint64_t MAX_CLUSTER_NUM_USER =
    DIMX * DIMY / ((CLUSTER_SIZE + 1) * (CLUSTER_SIZE + 1));

// a struct for the frame header
struct FrameHeader {
    std::uint64_t frameNumber;
    std::uint64_t bunchId;
};

// a struct for the frames
template <typename TData> struct Frame {
    FrameHeader header;
    TData data[DIMX * DIMY];
};

// the struct for the initial pedestal data
struct InitPedestal {
    std::size_t count;
    double oldM;
    double mean;
    double sigma;
    double stddev;
};

// execution flags to select the various kernels
struct ExecutionFlags {
    // 0 = only energy output, 1 = photon (and energy) output, 2 = clustering
    // (and energy) output
    uint8_t mode : 2;
    // 0 = off, 1 = on
    uint8_t summation : 1;
    // 0 = off, 1 = on
    uint8_t masking : 1;
    // 0 = off, 1 = on
    uint8_t maxValue : 1;
};

// a struct to hold one cluster
struct Cluster {
    std::uint64_t frameNumber;
    std::int16_t x;
    std::int16_t y;
    std::int32_t data[CLUSTER_SIZE * CLUSTER_SIZE];
};

// a struct to hold multiple clusters (on host and device)
template <typename TAlpaka, typename TDim, typename TSize> struct ClusterArray {
    std::size_t used;
    alpaka::mem::buf::
        Buf<typename TAlpaka::DevHost, unsigned long long, TDim, TSize>
            usedPinned;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, Cluster, TDim, TSize>
        clusters;

    ClusterArray(std::size_t maxClusterCount = MAX_CLUSTER_NUM * DEV_FRAMES,
                 typename TAlpaka::DevHost host =
                     alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u))
        : used(0),
          usedPinned(
              alpaka::mem::buf::alloc<unsigned long long, TSize>(host,
                                                                 SINGLEMAP)),
          clusters(
              alpaka::mem::buf::alloc<Cluster, TSize>(host, maxClusterCount))
    {
        alpaka::mem::buf::prepareForAsyncCopy(usedPinned);
        alpaka::mem::buf::prepareForAsyncCopy(clusters);

        alpaka::mem::view::getPtrNative(usedPinned)[0] = used;
    }
};

// a struct to hold multiple frames (on host and device)
template <typename T, typename TAlpaka, typename TDim, typename TSize>
struct FramePackage {
    std::size_t numFrames;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, T, TDim, TSize> data;

    FramePackage(std::size_t numFrames,
                 typename TAlpaka::DevHost host =
                     alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u))
        : numFrames(numFrames),
          data(alpaka::mem::buf::alloc<T, TSize>(host, numFrames))
    {
        alpaka::mem::buf::prepareForAsyncCopy(data);
    }
};

// type definitions
using Pedestal = double;
using EnergyValue = double;
using DetectorData = Frame<std::uint16_t>;
using PhotonMap = DetectorData;
using SumMap = Frame<double>;
using DriftMap = Frame<double>;
using GainStageMap = Frame<char>;
using MaskMap = Frame<bool>;
using EnergyMap = Frame<EnergyValue>;
using GainMap = CheapArray<double, DIMX * DIMY>;
using PedestalMap = CheapArray<double, DIMX * DIMY>;
using InitPedestalMap = CheapArray<InitPedestal, DIMX * DIMY>;

// debug statements
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds ms;
static Clock::time_point t;

#ifdef NDEBUG
#include <iostream>

// empty print to end recursion
template <typename... TArgs> void printArgs() { std::cout << std::endl; }

// print one or more argument
template <typename TFirst, typename... TArgs>
void printArgs(TFirst first, TArgs... args)
{
    std::cout << first << " ";
    printArgs(args...);
}

// general debug print function
template <typename... TArgs>
void debugPrint(const char* file, unsigned int line, TArgs... args)
{
    std::cout << __FILE__ << "[" << __LINE__ << "]:\n\t"
              << (std::chrono::duration_cast<ms>((Clock::now() - t))).count()
              << " ms\n\t";
    printArgs(args...);
}

#define DEBUG(...)                                                             \
    debugPrint(__FILE__, __LINE__, ##__VA_ARGS__); //                   \
  //(std::cout << __FILE__ << "[" << __LINE__ << "]:\n\t"               \
  //             << (std::chrono::duration_cast<ms>((Clock::now() - t))).count() \
  //             << " ms\n\t" << msg << std::endl) //"\n")
#else
#define DEBUG(...)
#endif
