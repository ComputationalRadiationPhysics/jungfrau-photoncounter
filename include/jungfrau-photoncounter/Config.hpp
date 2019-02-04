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
constexpr std::size_t DEV_FRAMES = 1000;
constexpr std::size_t PEDEMAPS = 3;
constexpr std::size_t GAINMAPS = 3;
constexpr float BEAMCONST = 6.2;
constexpr float PHOTONCONST = (1. / 12.4);
constexpr int CLUSTER_SIZE = 3;

// derived settings
constexpr std::size_t MAPSIZE = DIMX * DIMY;
constexpr std::size_t SINGLEMAP = 1;
constexpr std::size_t MAXINT = std::numeric_limits<uint32_t>::max();
constexpr char MASKED_VALUE = 4;
constexpr uint64_t MAX_CLUSTER_NUM = (DIMX - CLUSTER_SIZE + 1) *
                                     (DIMY - CLUSTER_SIZE + 1) /
                                     ((CLUSTER_SIZE / 2) * (CLUSTER_SIZE / 2));

struct FrameHeader {
    std::uint64_t frameNumber;
    std::uint64_t bunchId;
};

template <typename TData> struct Frame {
    FrameHeader header;
    TData data[DIMX * DIMY];
};

struct Pedestal {
    std::size_t count;
    double oldM;
    double mean;
    double oldS;
    double newS;
    double stddev;
    double variance;
};

struct Cluster {
    std::uint64_t frameNumber;
    std::int16_t x;
    std::int16_t y;
    std::int32_t data[CLUSTER_SIZE * CLUSTER_SIZE];
};

struct ExecutionFlags {
  // 0 = only energy output, 1 = photon (and energy) output, 2 = clustering (and energy) output
  uint8_t mode : 2;
  // 0 = off, 1 = on
  uint8_t summation : 1;
  // 0 = off, 1 = on
  uint8_t masking : 1;
  // 0 = off, 1 = on
  uint8_t maxValue : 1;
};

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
        alpaka::mem::buf::pin(usedPinned);
        alpaka::mem::buf::pin(clusters);

        alpaka::mem::view::getPtrNative(usedPinned)[0] = used;
    }
};

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
        alpaka::mem::buf::pin(data);
    }
};

using EnergyValue = double;
using DetectorData = Frame<std::uint16_t>;
using PhotonMap = DetectorData;
using EnergySumMap = Frame<std::uint64_t>;
using DriftMap = Frame<double>;
using GainStageMap = Frame<char>;
using MaskMap = Frame<bool>;
using EnergyMap = Frame<EnergyValue>;
using GainMap = CheapArray<double, DIMX * DIMY>;
using PedestalMap = CheapArray<Pedestal, DIMX * DIMY>;

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
               << " ms\n\t" << msg << "\n")
#else
#define DEBUG(msg)
#endif

template <typename TAlpaka, typename TDim, typename TSize>
void saveClusters(std::string path,
                  ClusterArray<TAlpaka, TDim, TSize>& clusters)
{
#if (SHOW_DEBUG)
    std::ofstream clusterFile;
    clusterFile.open(path);
    clusterFile << clusters.used << "\n";
    Cluster* clusterPtr = alpaka::mem::view::getPtrNative(clusters.clusters);

    DEBUG("writing " << clusters.used << " clusters to " << path);

    for (uint64_t i = 0; i < clusters.used; ++i) {
        // write cluster information
        clusterFile << static_cast<uint32_t>(clusterPtr[i].frameNumber &
                                             0xFFFFFFFF)
                    << "\n\t" << clusterPtr[i].x << " " << clusterPtr[i].y
                    << "\n";

        // write cluster
        for (uint8_t y = 0; y < CLUSTER_SIZE; ++y) {
            clusterFile << "\t";
            for (uint8_t x = 0; x < CLUSTER_SIZE; ++x) {
                clusterFile << clusterPtr[i].data[x + y * CLUSTER_SIZE] << " ";
            }

            clusterFile << "\n";
        }
    }

    clusterFile.close();
#endif
}

template <typename TBuffer>
void save_image(std::string path, TBuffer* data, std::size_t frame_number)
{
#if (SHOW_DEBUG)
    std::ofstream img;
    img.open(path + ".txt");
    for (std::size_t j = 0; j < 512; j++) {
        for (std::size_t k = 0; k < 1024; k++) {
            double h = double(data[frame_number].data[(j * 1024) + k]);
            img << h << " ";
        }
        img << "\n";
    }
    img.close();
#endif
}


template <typename TAlpaka, typename TDim, typename TSize>
void saveClusterArray(std::string path,
                      std::vector<ClusterArray<TAlpaka, TDim, TSize>>& clusters)
{
#if (SHOW_DEBUG)
    std::ofstream clusterFile;
    clusterFile.open(path);

    uint64_t numClusters = 0;
    for (const auto& clusterArray : clusters)
        numClusters += clusterArray.used;

    clusterFile << numClusters << "\n";

    DEBUG("writing " << numClusters << " clusters to " << path);

    for (auto& clusterArray : clusters) {
        Cluster* clusterPtr =
            alpaka::mem::view::getPtrNative(clusterArray.clusters);

        for (uint64_t i = 0; i < clusterArray.used; ++i) {
            // write cluster information
            clusterFile << static_cast<int32_t>(clusterPtr[i].frameNumber &
                                                0xFFFFFFFF)
                        << "\n\t" << clusterPtr[i].x << "\n\t"
                        << clusterPtr[i].y << "\n";

            // write cluster
            for (uint8_t y = 0; y < CLUSTER_SIZE; ++y) {
                clusterFile << "\t";
                for (uint8_t x = 0; x < CLUSTER_SIZE; ++x) {
                    clusterFile << clusterPtr[i].data[x + y * CLUSTER_SIZE]
                                << " ";
                }

                clusterFile << "\n";
            }
        }
    }

    clusterFile.close();
#endif
}
