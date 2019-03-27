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
    typename TAlpaka::DevHost host;
    typename TAlpaka::DevAcc device;
    typename TAlpaka::Queue queue;
    typename TAlpaka::Event event;
    State state;

    // device maps
    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, DetectorData, Dim, Size>
        data;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, GainMap, Dim, Size> gain;

    alpaka::mem::buf::
        Buf<typename TAlpaka::DevAcc, InitPedestalMap, Dim, Size>
            initialPedestal;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, PedestalMap, Dim, Size>
        pedestal;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, MaskMap, Dim, Size> mask;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, DriftMap, Dim, Size>
        drift;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, GainStageMap, Dim, Size>
        gainStage;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, GainStageMap, Dim, Size>
        gainStageOutput;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, EnergyMap, Dim, Size>
        energy;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, EnergyMap, Dim, Size>
        maxValueMaps;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, EnergyValue, Dim, Size>
        maxValues;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, PhotonMap, Dim, Size>
        photon;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, SumMap, Dim, Size>
        sum;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, Cluster, Dim, Size>
        cluster;

    alpaka::mem::buf::
        Buf<typename TAlpaka::DevAcc, unsigned long long, Dim, Size>
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
          data(alpaka::mem::buf::alloc<DetectorData, Size>(device, numMaps)),
          gain(alpaka::mem::buf::alloc<GainMap, Size>(device, GAINMAPS)),
          pedestal(
              alpaka::mem::buf::alloc<PedestalMap, Size>(device, PEDEMAPS)),
          initialPedestal(
              alpaka::mem::buf::alloc<InitPedestalMap, Size>(device,
                                                              PEDEMAPS)),
          drift(alpaka::mem::buf::alloc<DriftMap, Size>(device, numMaps)),
          gainStage(
              alpaka::mem::buf::alloc<GainStageMap, Size>(device, numMaps)),
          gainStageOutput(
              alpaka::mem::buf::alloc<GainStageMap, Size>(device, SINGLEMAP)),
          maxValueMaps(
              alpaka::mem::buf::alloc<EnergyMap, Size>(device, numMaps)),
          maxValues(
              alpaka::mem::buf::alloc<EnergyValue, Size>(device, numMaps)),
          energy(alpaka::mem::buf::alloc<EnergyMap, Size>(device, numMaps)),
          mask(alpaka::mem::buf::alloc<MaskMap, Size>(device, SINGLEMAP)),
          photon(alpaka::mem::buf::alloc<PhotonMap, Size>(device, numMaps)),
          sum(alpaka::mem::buf::alloc<SumMap, Size>(
              device,
              (numMaps + SUM_FRAMES - 1) / SUM_FRAMES)),
          cluster(alpaka::mem::buf::alloc<Cluster, Size>(device,
                                                          MAX_CLUSTER_NUM *
                                                              numMaps)),
          numClusters(
              alpaka::mem::buf::alloc<unsigned long long, Size>(device,
                                                                 SINGLEMAP))
    {
        // pin all buffer
        alpaka::mem::buf::prepareForAsyncCopy(data);
        alpaka::mem::buf::prepareForAsyncCopy(gain);
        alpaka::mem::buf::prepareForAsyncCopy(initialPedestal);
        alpaka::mem::buf::prepareForAsyncCopy(pedestal);
        alpaka::mem::buf::prepareForAsyncCopy(mask);
        alpaka::mem::buf::prepareForAsyncCopy(drift);
        alpaka::mem::buf::prepareForAsyncCopy(gainStage);
        alpaka::mem::buf::prepareForAsyncCopy(energy);
        alpaka::mem::buf::prepareForAsyncCopy(maxValueMaps);
        alpaka::mem::buf::prepareForAsyncCopy(maxValues);
        alpaka::mem::buf::prepareForAsyncCopy(photon);
        alpaka::mem::buf::prepareForAsyncCopy(sum);
        alpaka::mem::buf::prepareForAsyncCopy(cluster);
        alpaka::mem::buf::prepareForAsyncCopy(numClusters);

        // set cluster counter to 0
        alpaka::mem::view::set(queue, numClusters, 0, SINGLEMAP);
    }
};
