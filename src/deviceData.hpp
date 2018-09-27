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
        maxValue;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, PhotonMap, TDim, TSize>
        photon;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc, PhotonSumMap, TDim, TSize>
        sum;

    // host maps
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, PhotonMap, TDim, TSize>
        photonHost;

    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, PhotonSumMap, TDim, TSize>
        sumHost;

    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, EnergyMap, TDim, TSize>
        maxValueHost;

    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, EnergyMap, TDim, TSize>
        energyHost;

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
          drift(alpaka::mem::buf::alloc<DriftMap, TSize>(device, numMaps)),
          gainStage(
              alpaka::mem::buf::alloc<GainStageMap, TSize>(device, numMaps)),
          gainStageOutput(
              alpaka::mem::buf::alloc<GainStageMap, TSize>(device, SINGLEMAP)),
          maxValue(alpaka::mem::buf::alloc<EnergyMap, TSize>(device, numMaps)),
          energy(alpaka::mem::buf::alloc<EnergyMap, TSize>(device, numMaps)),
          mask(alpaka::mem::buf::alloc<MaskMap, TSize>(device, SINGLEMAP)),
          photon(alpaka::mem::buf::alloc<PhotonMap, TSize>(device, numMaps)),
          sum(alpaka::mem::buf::alloc<PhotonSumMap, TSize>(device,
                                                           numMaps /
                                                               SUM_FRAMES)),
          maxValueHost(
              alpaka::mem::buf::alloc<EnergyMap, TSize>(host, SINGLEMAP)),
          photonHost(alpaka::mem::buf::alloc<PhotonMap, TSize>(host, numMaps)),
          sumHost(alpaka::mem::buf::alloc<PhotonSumMap, TSize>(host,
                                                               numMaps /
                                                                   SUM_FRAMES)),
          energyHost(alpaka::mem::buf::alloc<EnergyMap, TSize>(host, numMaps))
    {
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#if (SHOW_DEBUG == false)
        // pin all buffer
        alpaka::mem::buf::pin(devices[num].data);
        alpaka::mem::buf::pin(devices[num].gain);
        alpaka::mem::buf::pin(devices[num].pedestal);
        alpaka::mem::buf::pin(devices[num].mask);
        alpaka::mem::buf::pin(devices[num].drift);
        alpaka::mem::buf::pin(devices[num].gainStage);
        alpaka::mem::buf::pin(devices[num].energy);
        alpaka::mem::buf::pin(devices[num].maxValue);
        alpaka::mem::buf::pin(devices[num].photon);
        alpaka::mem::buf::pin(devices[num].sum);
        alpaka::mem::buf::pin(devices[num].photonHost);
        alpaka::mem::buf::pin(devices[num].sumHost);
        alpaka::mem::buf::pin(devices[num].maxValueHost);
        alpaka::mem::buf::pin(devices[num].energyHost);
        alpaka::mem::buf::pin(devices[num].gainStageHost);
#endif
#endif
    }
};
