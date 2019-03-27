#pragma once

#include <alpaka/alpaka.hpp>

#include "Config.hpp"

enum State { FREE, PROCESSING, READY };

/**
 * This class manages the upload and download of data packages to all
 * devices. It's fully templated to use one of the structs provided
 * by Alpakaconfig.hpp.
 */
template <typename TAlpaka> struct DeviceData {
    std::size_t id;
    std::size_t numMaps;
    typename TAlpaka::DevAcc device;
    typename TAlpaka::Queue queue;
    typename TAlpaka::Event event;
    State state;

    // device maps
    typename TAlpaka::AccBuf<DetectorData> data;
    typename TAlpaka::AccBuf<GainMap> gain;
    typename TAlpaka::AccBuf<InitPedestalMap> initialPedestal;
    typename TAlpaka::AccBuf<PedestalMap> pedestal;
    typename TAlpaka::AccBuf<MaskMap> mask;
    typename TAlpaka::AccBuf<DriftMap> drift;
    typename TAlpaka::AccBuf<GainStageMap> gainStage;
    typename TAlpaka::AccBuf<GainStageMap> gainStageOutput;
    typename TAlpaka::AccBuf<EnergyMap> energy;
    typename TAlpaka::AccBuf<EnergyMap> maxValueMaps;
    typename TAlpaka::AccBuf<EnergyValue> maxValues;
    typename TAlpaka::AccBuf<PhotonMap> photon;
    typename TAlpaka::AccBuf<SumMap> sum;
    typename TAlpaka::AccBuf<Cluster> cluster;
    typename TAlpaka::AccBuf<unsigned long long> numClusters;

    DeviceData(std::size_t id,
               typename TAlpaka::DevAcc device,
               std::size_t numMaps = DEV_FRAMES)
        : id(id),
          numMaps(numMaps),
          device(device),
          queue(device),
          event(device),
          state(FREE),
          data(alpakaAlloc<DetectorData>(device, numMaps)),
          gain(alpakaAlloc<GainMap>(device, GAINMAPS)),
          pedestal(alpakaAlloc<PedestalMap>(device, PEDEMAPS)),
          initialPedestal(alpakaAlloc<InitPedestalMap>(device, PEDEMAPS)),
          drift(alpakaAlloc<DriftMap>(device, numMaps)),
          gainStage(alpakaAlloc<GainStageMap>(device, numMaps)),
          gainStageOutput(alpakaAlloc<GainStageMap>(device, SINGLEMAP)),
          maxValueMaps(alpakaAlloc<EnergyMap>(device, numMaps)),
          maxValues(alpakaAlloc<EnergyValue>(device, numMaps)),
          energy(alpakaAlloc<EnergyMap>(device, numMaps)),
          mask(alpakaAlloc<MaskMap>(device, SINGLEMAP)),
          photon(alpakaAlloc<PhotonMap>(device, numMaps)),
          sum(alpakaAlloc<SumMap>(device,
                                              (numMaps + SUM_FRAMES - 1) /
                                                  SUM_FRAMES)),
          cluster(alpakaAlloc<Cluster>(device,
                                                   MAX_CLUSTER_NUM * numMaps)),
          numClusters(alpakaAlloc<unsigned long long>(device, SINGLEMAP))
    {
        // set cluster counter to 0
        alpakaMemSet(queue, numClusters, 0, SINGLEMAP);
    }
};
