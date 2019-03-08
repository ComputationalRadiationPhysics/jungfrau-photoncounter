#pragma once

#include <alpaka/alpaka.hpp>

#include "Config.hpp"

enum State { FREE, PROCESSING, READY };

/**
 * This class manages the upload and download of data packages to all
 * devices. It's fully templated to use one of the structs provided
 * by Alpakaconfig.hpp.
 */
template <typename TAlpaka, typename TDim, typename TSize> struct DeviceData {
    std::size_t id;
    std::size_t numMaps; //! @todo: is this ever used?
    typename TAlpaka::DevHost host;
    typename TAlpaka::DevAcc device;
    typename TAlpaka::Queue queue;
    typename TAlpaka::Event event;
    State state;

    // device maps
    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, DetectorData, TDim, TSize>
        data;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, GainMap, TDim, TSize> gain;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, PedestalMap, TDim, TSize>
        initialPedestal;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, PedestalMap, TDim, TSize>
        pedestal;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, MaskMap, TDim, TSize> mask;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, DriftMap, TDim, TSize>
        drift;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, GainStageMap, TDim, TSize>
        gainStage;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, GainStageMap, TDim, TSize>
        gainStageOutput;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, EnergyMap, TDim, TSize>
        energy;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, EnergyMap, TDim, TSize>
        maxValueMaps;
  
    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, EnergyValue, TDim, TSize>
        maxValues;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, PhotonMap, TDim, TSize>
        photon;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, SumMap, TDim, TSize>
        sum;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, Cluster, TDim, TSize>
        cluster;

    alpaka::mem::buf::
        Buf<typename TAlpaka::DevAcc, unsigned long long, TDim, TSize>
            numClusters;

    DeviceData(std::size_t id,
               typename TAlpaka::DevAcc device,
               std::size_t numMaps = DEV_FRAMES)
        : id(id),
          numMaps(numMaps),
          host(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u)),
          device(device),
          queue(device),
          event(device),
          state(FREE),
          data(alpaka::mem::buf::alloc<DetectorData, TSize>(device, numMaps)),
          gain(alpaka::mem::buf::alloc<GainMap, TSize>(device, GAINMAPS)),
          pedestal(
              alpaka::mem::buf::alloc<PedestalMap, TSize>(device, PEDEMAPS)),
          initialPedestal(
              alpaka::mem::buf::alloc<PedestalMap, TSize>(device, PEDEMAPS)),
          drift(alpaka::mem::buf::alloc<DriftMap, TSize>(device, numMaps)),
          gainStage(
              alpaka::mem::buf::alloc<GainStageMap, TSize>(device, numMaps)),
          gainStageOutput(
              alpaka::mem::buf::alloc<GainStageMap, TSize>(device, SINGLEMAP)),
          maxValueMaps(alpaka::mem::buf::alloc<EnergyMap, TSize>(device, numMaps)),
          maxValues(alpaka::mem::buf::alloc<EnergyValue, TSize>(device, numMaps)),
          energy(alpaka::mem::buf::alloc<EnergyMap, TSize>(device, numMaps)),
          mask(alpaka::mem::buf::alloc<MaskMap, TSize>(device, SINGLEMAP)),
          photon(alpaka::mem::buf::alloc<PhotonMap, TSize>(device, numMaps)),
          sum(alpaka::mem::buf::alloc<SumMap, TSize>(device,
                                                           numMaps /
                                                               SUM_FRAMES)),
          cluster(alpaka::mem::buf::alloc<Cluster, TSize>(device,
                                                          MAX_CLUSTER_NUM *
                                                              numMaps)),
          numClusters(
              alpaka::mem::buf::alloc<unsigned long long, TSize>(device,
                                                                 SINGLEMAP))
    {
        // pin all buffer
        alpaka::mem::buf::pin(data);
        alpaka::mem::buf::pin(gain);
        alpaka::mem::buf::pin(initialPedestal);
        alpaka::mem::buf::pin(pedestal);
        alpaka::mem::buf::pin(mask);
        alpaka::mem::buf::pin(drift);
        alpaka::mem::buf::pin(gainStage);
        alpaka::mem::buf::pin(energy);
        alpaka::mem::buf::pin(maxValueMaps);
        alpaka::mem::buf::pin(maxValues);
        alpaka::mem::buf::pin(photon);
        alpaka::mem::buf::pin(sum);
        alpaka::mem::buf::pin(cluster);
        alpaka::mem::buf::pin(numClusters);

        // set cluster counter to 0
        alpaka::mem::view::set(queue, numClusters, 0, SINGLEMAP);
    }
};
