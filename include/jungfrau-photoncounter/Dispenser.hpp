#pragma once

#include "Alpakaconfig.hpp"

#include <iostream>
#include <chrono>

template <typename T, uint64_t size> struct CheapArray {
    T data[size];

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE T& operator[](uint64_t index)
    {
        return data[index];
    }

    ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE const T&
    operator[](uint64_t index) const
    {
        return data[index];
    }
};

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

// a struct to hold multiple frames (on host and device)
template <typename T, typename TAlpaka> struct FramePackage {
    std::size_t numFrames;
    typename TAlpaka::template HostBuf<T> data;

    FramePackage(std::size_t numFrames,
                 typename TAlpaka::DevHost host = alpakaGetHost<TAlpaka>())
        : numFrames(numFrames), data(alpakaAlloc<T>(host, numFrames))
    {
    }
};

// type definitions
using Pedestal = double;
using EnergyValue = double;

// a struct to hold all detector specific configuration variables
template <std::size_t TDimX,
          std::size_t TDimY,
          std::size_t TDevFrames,
          std::size_t TClusterSize>
struct DetectorConfig {
    static constexpr std::size_t DIMX = TDimX;
    static constexpr std::size_t DIMY = TDimY;
    static constexpr std::size_t DEV_FRAMES = TDevFrames;
    static constexpr std::size_t CLUSTER_SIZE = TClusterSize;

    // general settings
    static constexpr std::size_t FRAME_HEADER_SIZE = 16;
    static constexpr std::size_t PEDEMAPS = 3;
    static constexpr std::size_t GAINMAPS = 3;

    // derived settings
    static constexpr std::size_t MAPSIZE = DIMX * DIMY;
    static constexpr std::size_t SINGLEMAP = 1;

    // a struct for the frames
    template <typename TData> struct Frame {
        FrameHeader header;
        TData data[DIMX * DIMY];
    };

    using DetectorData = Frame<std::uint16_t>;
    using GainStageMap = Frame<char>;
    using MaskMap = Frame<bool>;
    using EnergyMap = Frame<EnergyValue>;
    using GainMap = CheapArray<double, DIMX * DIMY>;
    using PedestalMap = CheapArray<double, DIMX * DIMY>;
    using InitPedestalMap = CheapArray<InitPedestal, DIMX * DIMY>;
};

// debug statements
typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::milliseconds ms;
static Clock::time_point t;

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
    std::cout << file << "[" << line << "]:\n\t"
              << (std::chrono::duration_cast<ms>((Clock::now() - t))).count()
              << " ms\n\t";
    printArgs(args...);
}

#define DEBUG(...) debugPrint(__FILE__, __LINE__, ##__VA_ARGS__)


template <typename TAcc>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto getLinearIdx(const TAcc &acc)
    -> std::uint64_t {
  auto const globalThreadIdx = alpakaGetGlobalThreadIdx(acc);
  auto const globalThreadExtent = alpakaGetGlobalThreadExtent(acc);

  auto const linearizedGlobalThreadIdx =
      alpakaGetGlobalLinearizedGlobalThreadIdx(globalThreadIdx,
                                               globalThreadExtent);

  return linearizedGlobalThreadIdx[0u];
}

template <typename TConfig, typename TMap, typename TThreadIndex, typename TSum>
ALPAKA_FN_ACC ALPAKA_FN_INLINE auto
findClusterSumAndMax(TMap const &map, TThreadIndex const id, TSum &sum,
                     TThreadIndex &max) -> void {
  TThreadIndex it = 0;
  max = 0;
  // bug might be here
  //sum = 0; // when commenting in this line all accelerators give the same result
  constexpr int n = TConfig::CLUSTER_SIZE;
  for (int y = -n / 2; y < (n + 1) / 2; ++y) {
    for (int x = -n / 2; x < (n + 1) / 2; ++x) {
      it = id + y * TConfig::DIMX + x;
      if (map[max] < map[it])
        max = it;
      sum += map[it];
    }
  }
}


template <typename Config> struct GainmapInversionKernel {
  template <typename TAcc, typename TGain>
  ALPAKA_FN_ACC auto operator()(TAcc const &acc, TGain *const gainmaps) const
      -> void {
    auto globalId = getLinearIdx(acc);
    auto elementsPerThread = alpakaGetElementExtent(acc)[0u];
    
    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread; ++id) {
      // check range
      if (id >= Config::MAPSIZE)
        break;
      
     for (size_t i = 0; i < Config::GAINMAPS; ++i) {
	const auto invers =  1.0 / gainmaps[i][id];
        gainmaps[i][id] = invers;
      }
    }
  }
};




template <typename Config, typename AccConfig> struct ClusterEnergyKernel {
  template <typename TAcc, typename TDetectorData, typename TGainMap,
            typename TPedestalMap, typename TGainStageMap, typename TEnergyMap, typename TMask>
  ALPAKA_FN_ACC auto operator()(
      TAcc const &acc, TDetectorData const *const detectorData,
      TGainMap const *const gainMaps,
      TPedestalMap *const pedestalMaps, TGainStageMap *const gainStageMaps,
      TEnergyMap *const energyMaps, TMask *const mask) const -> void {
    const uint64_t globalId = getLinearIdx(acc);
    constexpr uint64_t elementsPerThread = AccConfig::elementsPerThread;

    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread &&
           id < Config::MAPSIZE; ++id) {

  	// use masks to check whether the channel is valid or masked out
  	bool isValid = !mask ? 1 : mask->data[id];

  	auto dataword = detectorData[0].data[id];
  	auto adc = dataword & 0x3fff;

  	auto &gainStage = gainStageMaps[0].data[id];
  	gainStage = (dataword & 0xc000) >> 14;
        if (gainStage == 3)
            gainStage = 2;

  	const auto &pedestal = pedestalMaps[gainStage][id];
  	const auto &gain = gainMaps[gainStage][id];

	// calculate energy of current channel
  	auto &energy = energyMaps[0].data[id];

  	energy = (adc - pedestal) * gain;

  	// set energy to zero if masked out
  	if (!isValid)
    	    energy = 0;
    }
  }
};

template <typename Config, typename AccConfig> struct ClusterFinderKernel {
  template <typename TAcc, typename TInitPedestalMap,
            typename TGainStageMap, typename TEnergyMap,
            typename TNumClusters, typename TMask,
            typename TNumStdDevs = int>
  ALPAKA_FN_ACC auto operator()(
      TAcc const &acc, TInitPedestalMap *const initPedestalMaps,
      TGainStageMap *const gainStageMaps,
      TEnergyMap *const energyMaps,
      TNumClusters *const numClusters, TMask *const mask) const -> void {
    const uint64_t globalId = getLinearIdx(acc);
    constexpr uint64_t elementsPerThread = AccConfig::elementsPerThread;

    constexpr auto n = Config::CLUSTER_SIZE;

    // iterate over all elements in the thread
    for (auto id = globalId * elementsPerThread;
         id < (globalId + 1) * elementsPerThread &&
           id < Config::MAPSIZE; ++id) {
      
      const char &gainStage = gainStageMaps->data[id];
      float sum;
      decltype(id) max;
      const auto &energy = energyMaps->data[id];
      const auto &stddev = initPedestalMaps[gainStage][id].stddev;
      if ((id % Config::DIMX >= n / 2 &&
          id % Config::DIMX <= Config::DIMX - (n + 1) / 2 &&
          id / Config::DIMX >= n / 2 &&
          id / Config::DIMX <= Config::DIMY - (n + 1) / 2)) {
        findClusterSumAndMax<Config>(energyMaps->data, id,
                                     sum, max);
        // check cluster conditions
        if ((energy > 5 * stddev || sum > n * 5 * stddev) && id == max) {
		alpakaAtomicAdd(acc, numClusters, static_cast<TNumClusters>(1));
        }
      }
    } 
  }
};











template <typename Config, typename TAlpaka> struct DeviceData {
    typename TAlpaka::DevAcc* device;
    typename TAlpaka::Queue queue;

    // device maps
    typename TAlpaka::template AccBuf<typename Config::DetectorData> data;
    typename TAlpaka::template AccBuf<typename Config::GainMap> gain;
    typename TAlpaka::template AccBuf<typename Config::InitPedestalMap>
        initialPedestal;
    typename TAlpaka::template AccBuf<typename Config::PedestalMap> pedestal;
    typename TAlpaka::template AccBuf<typename Config::MaskMap> mask;
    typename TAlpaka::template AccBuf<typename Config::GainStageMap> gainStage;
    typename TAlpaka::template AccBuf<typename Config::EnergyMap> energy;
    typename TAlpaka::template AccBuf<unsigned long long> numClusters;

    DeviceData(typename TAlpaka::DevAcc* devPtr)
        : device(devPtr),
          queue(*device),
          data(alpakaAlloc<typename Config::DetectorData>(
              *device,
              decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
          gain(alpakaAlloc<typename Config::GainMap>(
              *device,
              decltype(Config::GAINMAPS)(Config::GAINMAPS))),
          pedestal(alpakaAlloc<typename Config::PedestalMap>(
              *device,
              decltype(Config::PEDEMAPS)(Config::PEDEMAPS))),
          initialPedestal(alpakaAlloc<typename Config::InitPedestalMap>(
              *device,
              decltype(Config::PEDEMAPS)(Config::PEDEMAPS))),
          gainStage(alpakaAlloc<typename Config::GainStageMap>(
              *device,
              decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
          energy(alpakaAlloc<typename Config::EnergyMap>(
              *device,
              decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
          mask(alpakaAlloc<typename Config::MaskMap>(
              *device,
              decltype(Config::SINGLEMAP)(Config::SINGLEMAP))),
          numClusters(alpakaAlloc<unsigned long long>(
              *device,
              decltype(Config::SINGLEMAP)(Config::SINGLEMAP )))
    {
        // set cluster counter to 0
        alpakaMemSet(queue,
                     numClusters,
                     0,
                     decltype(Config::SINGLEMAP)(Config::SINGLEMAP));
    }
};



template <typename TConfig, template <std::size_t> typename TAccelerator>
class Dispenser {
public:
    // use types defined in the config struct
    using TAlpaka = TAccelerator<TConfig::MAPSIZE>;

    Dispenser(FramePackage<typename TConfig::GainMap, TAlpaka> gain,
              FramePackage<typename TConfig::MaskMap, TAlpaka> mask,
	      FramePackage<typename TConfig::InitPedestalMap, TAlpaka> initialPedestals, 
	      FramePackage<typename TConfig::PedestalMap, TAlpaka> pedestals)
        : device(&alpakaGetDevs<TAlpaka>()[0])
    {
        const GainmapInversionKernel<TConfig> gainmapInversionKernel{};

        alpakaCopy(device.queue,
                   device.gain,
                   gain.data,
                   decltype(TConfig::GAINMAPS)(TConfig::GAINMAPS));
            
        // compute reciprocals of gain maps
        auto const gainmapInversion(alpakaCreateKernel<TAlpaka>(getWorkDiv<TAlpaka>(),
                                                                gainmapInversionKernel,
                                                                alpakaNativePtr(device.gain)));
        alpakaEnqueueKernel(device.queue, gainmapInversion);

        // wait until everything is finished
        alpakaWait(device.queue);

        DEBUG("Loading existing mask map on device", 0);
        alpakaCopy(device.queue,
                   device.mask,
                   mask.data,
                   decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

        DEBUG("distribute raw pedestal maps");
        alpakaCopy(device.queue,
                   device.initialPedestal,
                   initialPedestals.data,
                   decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));
        alpakaCopy(device.queue,
                   device.pedestal,
                   pedestals.data,
                   decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

        alpakaWait(device.queue);

        DEBUG("Device #", 0, "initialized!");
    }

  template <typename TFramePackageEnergyMap>
    auto process(typename TAlpaka::template HostBuf<typename TConfig::DetectorData> data,
                 std::size_t offset,
                 TFramePackageEnergyMap energy,
                 FramePackage<typename TConfig::EnergyMap, TAlpaka > energydata,
                 FramePackage<typename TConfig::GainStageMap, TAlpaka> gainstagedata)
        -> void {

        // upload input data
        alpakaCopy(device.queue,
                   device.data,
                   data,
                   1);
        

        // clustering (and conversion to energy)
        DEBUG("enqueueing clustering kernel");
                
        // reset the number of clusters
        alpakaMemSet(device.queue,
                     device.numClusters,
                     0,
                     decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));
              
          ClusterEnergyKernel<TConfig, TAlpaka> clusterEnergyKernel{};
          auto const clusterEnergy(alpakaCreateKernel<TAlpaka>(getWorkDiv<TAlpaka>(),
                                                               clusterEnergyKernel,
                                                               alpakaNativePtr(device.data),
                                                               alpakaNativePtr(device.gain),
                                                               alpakaNativePtr(device.pedestal),
                                                               alpakaNativePtr(device.gainStage),
                                                               alpakaNativePtr(device.energy),
                                                               alpakaNativePtr(device.mask)));

          alpakaEnqueueKernel(device.queue, clusterEnergy);
          alpakaWait(device.queue);


          /*        
        // upload energy data
        alpakaCopy(device.queue,
                   device.energy,
                   energydata.data,
                   numMaps);
        
        // upload gainstage data
        alpakaCopy(device.queue,
                   device.gainStage,
                   gainstagedata.data,
                   numMaps);
          */


          
          // execute cluster finder
          ClusterFinderKernel<TConfig, TAlpaka> clusterFinderKernel{};
          auto const clusterFinder(alpakaCreateKernel<TAlpaka>(getWorkDiv<TAlpaka>(),
                                                               clusterFinderKernel,
                                                               alpakaNativePtr(device.initialPedestal),
                                                               alpakaNativePtr(device.gainStage),
                                                               alpakaNativePtr(device.energy),
                                                               alpakaNativePtr(device.numClusters),
                                                               alpakaNativePtr(device.mask)));

          alpakaEnqueueKernel(device.queue, clusterFinder);
          alpakaWait(device.queue);

	  typename TAlpaka::template HostBuf<unsigned long long> numClusters = alpakaAlloc<unsigned long long>(alpakaGetHost<TAlpaka>(), (std::size_t)1);


            alpakaCopy(device.queue,
                       numClusters,
                       device.numClusters,
                       1);
            
            alpakaWait(device.queue);
            auto clustersToDownload = alpakaNativePtr(numClusters)[0];

            DEBUG("Downloading ",
                  clustersToDownload);

        alpakaWait(device.queue);
    }

private:
  DeviceData<TConfig, TAlpaka> device;
};
