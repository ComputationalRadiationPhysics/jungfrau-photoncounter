#pragma once

#include <alpaka/alpaka.hpp>
#include <chrono>
#include <fstream>
#include <iostream>

#include <optional.hpp>

#include "AlpakaHelper.hpp"
#include "CheapArray.hpp"

// debug statements
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds ms;
static Clock::time_point t;

//#ifdef VERBOSE
#include <iostream>

// empty print to end recursion
template <typename... TArgs> void printArgs() { std::cerr << std::endl; }

// print one or more argument
template <typename TFirst, typename... TArgs>
void printArgs(TFirst first, TArgs... args) {
  std::cerr << first << " ";
  printArgs(args...);
}

// general debug print function
template <typename... TArgs>
void debugPrint(const char *file, unsigned int line, TArgs... args) {
  std::cerr << file << "[" << line << "]:\n\t"
            << (std::chrono::duration_cast<ms>((Clock::now() - t))).count()
            << " ms\n\t";
  printArgs(args...);
}

#define DEBUG(...) debugPrint(__FILE__, __LINE__, ##__VA_ARGS__)
//#else
//#define DEBUG(...)
//#endif

// a struct for the frame header
struct FrameHeader {
  std::uint64_t frameNumber;
  std::uint64_t bunchId;
};

// the struct for the initial pedestal data
struct InitPedestal {
  std::size_t count;
  double mean;
  double m;
  double m2;
  double stddev;
};

// execution flags to select the various kernels
struct ExecutionFlags {
  // 0 = only energy output, 1 = photon (and energy) output, 2 =
  // clustering (and energy) output, 3 = clustering and explicit energy
  // output (used in benchmarks)
  uint8_t mode : 2;
  // 0 = off, 1 = on
  uint8_t summation : 1;
  // 0 = off, 1 = on
  uint8_t masking : 1;
  // 0 = off, 1 = on
  uint8_t maxValue : 1;
};

// a struct to hold a view of multiple frames (on host and device)
template <typename T, typename TAlpaka, typename TBufView>
struct FramePackageView {
  std::size_t numFrames;
  typename TAlpaka::template HostView<T> data;

  FramePackageView(TBufView data, std::size_t offset, std::size_t numFrames)
      : numFrames(numFrames), data(data, numFrames, offset) {}

  FramePackageView<T, TAlpaka, typename TAlpaka::template HostView<T>>
  getView(std::size_t offset, std::size_t numFrames) {
    return FramePackageView<T, TAlpaka, typename TAlpaka::template HostView<T>>(
        data, offset, numFrames);
  }
};

// a struct to hold multiple frames (on host and device)
template <typename T, typename TAlpaka> struct FramePackage {
  std::size_t numFrames;
  typename TAlpaka::template HostBuf<T> data;

  FramePackage(std::size_t numFrames,
               typename TAlpaka::DevHost host = alpakaGetHost<TAlpaka>())
      : numFrames(numFrames), data(alpakaAlloc<T>(host, numFrames)) {}

  FramePackageView<T, TAlpaka, typename TAlpaka::template HostView<T>>
  getView(std::size_t offset, std::size_t numFrames) {
    return FramePackageView<T, TAlpaka, typename TAlpaka::template HostView<T>>(
        static_cast<typename TAlpaka::template HostView<T>>(data), offset,
        numFrames);
  }
};
// define void_t
template <typename... Ts> struct make_void { typedef void type; };
template <typename... Ts> using void_t = typename make_void<Ts...>::type;

// define a double buffer / shadow buffer
template <typename T, typename TAccelerator, typename TBufferHost,
          typename TBufferDevice, bool>
class DoubleBuffer;

// CPU implementation
template <typename T, typename TAccelerator, typename TBufferHost,
          typename TBufferDevice>
class DoubleBuffer<T, TAccelerator, TBufferHost, TBufferDevice, true> {
public:
  // construct from optional host buffer and normal device buffer
  DoubleBuffer(tl::optional<TBufferHost> h, TBufferDevice d,
               typename TAccelerator::Queue *, std::size_t numMaps)
      : h(h ? getView(*h, numMaps) : getView(d, numMaps)) {}

  // create from normal host- and device buffer
  DoubleBuffer(TBufferHost h, TBufferDevice d, typename TAccelerator::Queue *,
               std::size_t numMaps)
      : h(getView(h, numMaps)) {}

  void resize(std::size_t) {}

  void upload() {}

  void download() {}

  // get the device buffer
  typename TAccelerator::template HostView<T> get() { return h; }

private:
  //! @todo: is this really required?
  // extract the view from a FramePackage
  static typename TAccelerator::template HostView<T>
  getView(FramePackage<T, TAccelerator> buf, std::size_t numMaps) {
    (void)numMaps;
    return getView(buf.data);
  }

  //! @todo: is this really required?
  // extract the view from a FramePackageView
  static typename TAccelerator::template HostView<T>
  getView(FramePackageView<T, TAccelerator,
                           typename TAccelerator::template HostView<T>>
              buf,
          std::size_t numMaps) {
    (void)numMaps;
    return buf.data;
  }

  // extract the view from a normal buffer
  static typename TAccelerator::template HostView<T>
  getView(typename TAccelerator::template HostBuf<T> buf, std::size_t numMaps) {
    long unsigned int offset = 0;
    return typename TAccelerator::template HostView<T>(buf, numMaps, offset);
  }

  // extract the view from another view
  static typename TAccelerator::template HostView<T>
  getView(typename TAccelerator::template HostView<T> view,
          std::size_t numMaps) {
    return view;
  }

  typename TAccelerator::template HostView<T> h;
};

// GPU implementation
template <typename T, typename TAccelerator, typename TBufferHost,
          typename TBufferDevice>
class DoubleBuffer<T, TAccelerator, TBufferHost, TBufferDevice, false> {
public:
  // construct from optional host buffer and normal device buffer
  DoubleBuffer(tl::optional<TBufferHost> h, TBufferDevice d,
               typename TAccelerator::Queue *queue, std::size_t numMaps)
      : queue(queue), h(h), d(d), numMaps(numMaps) {}

  // create from normal host- and device buffer
  DoubleBuffer(TBufferHost h, TBufferDevice d,
               typename TAccelerator::Queue *queue, std::size_t numMaps)
      : queue(queue), h(h), d(d), numMaps(numMaps) {}

  // upload to GPU
  void upload() {
    //! @todo: print error message if nothing is uploaded???
    if (h)
      alpakaCopy(*queue, d, *h, numMaps);
  }

  // download from GPU
  void download() {
    if (h) {
      DEBUG("start download", numMaps, "");
      alpakaCopy(*queue, *h, d, numMaps);
      DEBUG("end download");
    }
  }

  void resize(std::size_t numMaps) { this->numMaps = numMaps; }

  // get the GPU buffer
  typename TAccelerator::template AccView<T> get() { return d; }

private:
  typename TAccelerator::Queue *queue;
  tl::optional<TBufferHost> h;
  typename TAccelerator::template AccView<T> d;
  std::size_t numMaps;
};

// meta function to get the right class for the device
template <typename TAccelerator, typename TBufferHost, typename TBufferDevice>
struct GetDoubleBuffer {
  using Buffer = DoubleBuffer<
      typename alpaka::traits::ElemType<TBufferDevice>::type,
      TAccelerator, TBufferHost, TBufferDevice,
      std::is_same<alpaka::Dev<typename TAccelerator::Acc>,
                   alpaka::Dev<typename TAccelerator::Host>>::value>;
};

// extend the double buffer to FramePackages
template <typename TAccelerator, typename TBufferHost, typename TBufferDevice>
class FramePackageDoubleBuffer
    : public GetDoubleBuffer<TAccelerator, decltype(TBufferHost::data),
                             TBufferDevice>::Buffer {
public:
  FramePackageDoubleBuffer(tl::optional<TBufferHost> h, TBufferDevice d,
                           typename TAccelerator::Queue *queue,
                           std::size_t numMaps)
      : GetDoubleBuffer<TAccelerator, decltype(TBufferHost::data),
                        TBufferDevice>::
            Buffer(h ? static_cast<tl::optional<decltype(TBufferHost::data)>>(
                           h->data)
                     : static_cast<tl::optional<decltype(TBufferHost::data)>>(
                           tl::nullopt),
                   d, queue, numMaps) {}

  FramePackageDoubleBuffer(TBufferHost h, TBufferDevice d,
                           typename TAccelerator::Queue *queue,
                           std::size_t numMaps)
      : GetDoubleBuffer<TAccelerator, decltype(TBufferHost::data),
                        TBufferDevice>::Buffer(h.data, d, queue, numMaps) {}
};

// type definitions
using Pedestal = double;
using EnergyValue = double;
template <typename T, typename TAlpaka>
using FramePackageView_t =
    FramePackageView<T, TAlpaka, typename TAlpaka::template HostView<T>>;

// a struct to hold all detector specific configuration variables
template <std::size_t TFramesPerStageG0, std::size_t TFramesPerStageG1,
          std::size_t TFramesPerStageG2, std::size_t TDimX, std::size_t TDimY,
          std::size_t TSumFrames, std::size_t TDevFrames,
          std::size_t TMovingStatWindowSize, std::size_t TClusterSize,
          std::size_t TC>
struct DetectorConfig {
  static constexpr std::size_t FRAMESPERSTAGE_G0 = TFramesPerStageG0;
  static constexpr std::size_t FRAMESPERSTAGE_G1 = TFramesPerStageG1;
  static constexpr std::size_t FRAMESPERSTAGE_G2 = TFramesPerStageG2;
  static constexpr std::size_t DIMX = TDimX;
  static constexpr std::size_t DIMY = TDimY;
  static constexpr std::size_t SUM_FRAMES = TSumFrames;
  static constexpr std::size_t DEV_FRAMES = TDevFrames;
  static constexpr std::size_t MOVING_STAT_WINDOW_SIZE = TMovingStatWindowSize;
  static constexpr std::size_t CLUSTER_SIZE = TClusterSize;
  static constexpr std::size_t C = TC;

  // general settings
  static constexpr std::size_t FRAME_HEADER_SIZE = 16;
  static constexpr std::size_t PEDEMAPS = 3;
  static constexpr std::size_t GAINMAPS = 3;

  // derived settings
  static constexpr std::size_t MAPSIZE = DIMX * DIMY;
  static constexpr std::size_t SINGLEMAP = 1;
  static constexpr std::size_t MAXINT = std::numeric_limits<uint32_t>::max();
  static constexpr char MASKED_VALUE = 4;

  // maximal number of clusters possible:
  static constexpr uint64_t MAX_CLUSTER_NUM =
      (DIMX - CLUSTER_SIZE + 1) * (DIMY - CLUSTER_SIZE + 1) /
      ((CLUSTER_SIZE / 2) * (CLUSTER_SIZE / 2));

  static_assert(
      FRAMESPERSTAGE_G0 >= MOVING_STAT_WINDOW_SIZE,
      "Moving stat window size is bigger than the frames supplied for the "
      "callibration of the pedestal values for the first gain stage. ");

  // restrict number of clusters centers to 10% of all pixels (or the maximum
  // number of clusters)
  static constexpr uint64_t MAX_CLUSTER_NUM_USER =
      std::min(DIMX * DIMY / 10, MAX_CLUSTER_NUM);

  // maximal number of seperated clusters:
  // static constexpr uint64_t MAX_CLUSTER_NUM_USER = DIMX * DIMY /
  // ((CLUSTER_SIZE + 1) * (CLUSTER_SIZE + 1));

  // a struct for the frames
  template <typename TData> struct Frame {
    FrameHeader header;
    TData data[DIMX * DIMY];
  };

  // a struct to hold one cluster
  struct Cluster {
    std::uint64_t frameNumber;
    std::int16_t x;
    std::int16_t y;
    EnergyValue data[CLUSTER_SIZE * CLUSTER_SIZE];
  };

  // a struct to hold multiple clusters (on host and device)
  template <typename TAlpaka> struct ClusterArray {
    std::size_t used;
    typename TAlpaka::template HostBuf<unsigned long long> usedPinned;
    typename TAlpaka::template HostBuf<Cluster> clusters;

    ClusterArray(std::size_t maxClusterCount = MAX_CLUSTER_NUM_USER *
                                               DEV_FRAMES,
                 typename TAlpaka::DevHost host = alpakaGetHost<TAlpaka>())
        : used(0), usedPinned(alpakaAlloc<unsigned long long>(
                       host, decltype(SINGLEMAP)(SINGLEMAP))),
          clusters(alpakaAlloc<Cluster>(host, maxClusterCount)) {
      alpakaNativePtr(usedPinned)[0] = used;
    }
  };

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
};

// predefine detector configurations
using JungfrauConfig =
    DetectorConfig<1000, 1000, 999, 1024, 512, 2, 1, 100, 2, 5>;
using MoenchConfig = DetectorConfig<1000, 0, 0, 400, 400, 10, 300, 100, 3, 5>;
