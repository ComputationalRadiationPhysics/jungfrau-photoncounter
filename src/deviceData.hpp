#pragma once

#include <alpaka/alpaka.hpp>

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
    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          DetectorData,
                          TDim,
                          TSize>
        data;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          GainMap,
                          TDim,
                          TSize>
        gain;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          PedestalMap,
                          TDim,
                          TSize>
        pedestal;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          MaskMap,
                          TDim,
                          TSize>
        mask;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          DriftMap,
                          TDim,
                          TSize>
        drift;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          GainStageMap,
                          TDim,
                          TSize>
        gainStage;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          EnergyMap,
                          TDim,
                          TSize>
        maxValue;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          PhotonMap,
                          TDim,
                          TSize>
        photon;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          PhotonSumMap,
                          TDim,
                          TSize>
        sum;

    // host maps
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          PhotonMap,
                          TDim,
                          TSize>
        photonHost;

    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          PhotonSumMap,
                          TDim,
                          TSize>
        sumHost;

    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          EnergyMap,
                          TDim,
                          TSize>
        maxValueHost;

    DeviceData()
        : id(0),
          numMaps(0),
          host(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u)),
          device(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfAcc>(0u)),
          queue(device),
          event(device),
          state(FREE),
          data(alpaka::mem::buf::alloc<DetectorData, TSize>(device,
                                                                     0lu)),
          gain(alpaka::mem::buf::alloc<GainMap, TSize>(device,
                                                                     0lu)),
          pedestal(
              alpaka::mem::buf::alloc<PedestalMap, TSize>(device,
                                                                        0lu)),
          drift(alpaka::mem::buf::alloc<DriftMap, TSize>(device,
                                                                       0lu)),
          gainStage(
              alpaka::mem::buf::alloc<GainStageMap, TSize>(device,
                                                                         0lu)),
          maxValue(
              alpaka::mem::buf::alloc<EnergyMap, TSize>(device,
                                                                     0lu)),
          mask(alpaka::mem::buf::alloc<MaskMap, TSize>(device,
                                                                     0lu)),
          photon(alpaka::mem::buf::alloc<PhotonMap, TSize>(device,
                                                                         0lu)),
          sum(alpaka::mem::buf::alloc<PhotonSumMap, TSize>(device,
                                                                         0lu)),
          maxValueHost(
              alpaka::mem::buf::alloc<EnergyMap, TSize>(host,
                                                                     0lu)),
          photonHost(
              alpaka::mem::buf::alloc<PhotonMap, TSize>(host,
                                                                      0lu)),
          sumHost(
              alpaka::mem::buf::alloc<PhotonSumMap, TSize>(host,
                                                                         0lu))
    {
    }
};
