#pragma once

#include <alpaka/alpaka.hpp>
#include <future>
#include <memory>

#include "Config.hpp"

enum State { FREE, PROCESSING, READY };

/**
 * This class manages the upload and download of data packages to all
 * devices. It's fully templated to use one of the structs provided
 * by Alpakaconfig.hpp.
 */
template <typename Config, typename TAlpaka> struct DeviceData {

  std::size_t id;
  std::size_t numMaps;
  typename TAlpaka::DevAcc *device;
  typename TAlpaka::Queue queue;
  typename TAlpaka::Event event;
  State state;

  // device maps
  typename TAlpaka::template AccBuf<typename Config::DetectorData> data;
  typename TAlpaka::template AccBuf<typename Config::GainMap> gain;
  typename TAlpaka::template AccBuf<typename Config::InitPedestalMap>
      initialPedestal;
  typename TAlpaka::template AccBuf<typename Config::PedestalMap> pedestal;
  typename TAlpaka::template AccBuf<typename Config::MaskMap> mask;
  typename TAlpaka::template AccBuf<typename Config::DriftMap> drift;
  typename TAlpaka::template AccBuf<typename Config::GainStageMap> gainStage;
  typename TAlpaka::template AccBuf<typename Config::GainStageMap>
      gainStageOutput;
  typename TAlpaka::template AccBuf<typename Config::EnergyMap> energy;
  typename TAlpaka::template AccBuf<typename Config::EnergyMap> maxValueMaps;
  typename TAlpaka::template AccBuf<EnergyValue> maxValues;
  typename TAlpaka::template AccBuf<typename Config::PhotonMap> photon;
  typename TAlpaka::template AccBuf<typename Config::SumMap> sum;
  typename TAlpaka::template AccBuf<typename Config::Cluster> cluster;
  typename TAlpaka::template AccBuf<unsigned long long> numClusters;
  std::future<void> clusterDownload;

  DeviceData(std::size_t id, typename TAlpaka::DevAcc *devPtr)
      : id(id), numMaps(0), device(devPtr), queue(*device), event(*device),
        state(FREE),
        data(alpakaAlloc<typename Config::DetectorData>(
            *device, decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
        gain(alpakaAlloc<typename Config::GainMap>(
            *device, decltype(Config::GAINMAPS)(Config::GAINMAPS))),
        pedestal(alpakaAlloc<typename Config::PedestalMap>(
            *device, decltype(Config::PEDEMAPS)(Config::PEDEMAPS))),
        initialPedestal(alpakaAlloc<typename Config::InitPedestalMap>(
            *device, decltype(Config::PEDEMAPS)(Config::PEDEMAPS))),
        drift(alpakaAlloc<typename Config::DriftMap>(
            *device, decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
        gainStage(alpakaAlloc<typename Config::GainStageMap>(
            *device, decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
        gainStageOutput(alpakaAlloc<typename Config::GainStageMap>(
            *device, decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
        maxValueMaps(alpakaAlloc<typename Config::EnergyMap>(
            *device, decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
        maxValues(alpakaAlloc<EnergyValue>(
            *device, decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
        energy(alpakaAlloc<typename Config::EnergyMap>(
            *device, decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
        mask(alpakaAlloc<typename Config::MaskMap>(
            *device, decltype(Config::SINGLEMAP)(Config::SINGLEMAP))),
        photon(alpakaAlloc<typename Config::PhotonMap>(
            *device, decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
        sum(alpakaAlloc<typename Config::SumMap>(
            *device, ((decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES) +
                       Config::SUM_FRAMES) -
                      1) /
                         Config::SUM_FRAMES)),
        cluster(alpakaAlloc<typename Config::Cluster>(
            *device, decltype(Config::MAX_CLUSTER_NUM_USER)(
                         Config::MAX_CLUSTER_NUM_USER) *
                         decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
        numClusters(alpakaAlloc<unsigned long long>(
            *device, decltype(Config::SINGLEMAP)(Config::SINGLEMAP))) {
    // set cluster counter to 0
    alpakaMemSet(queue, numClusters, 0,
                 decltype(Config::SINGLEMAP)(Config::SINGLEMAP));
  }
};
