#pragma once

#include <alpaka/alpaka.hpp>

enum State { FREE, PROCESSING, READY };

/**
 * This class manages the upload and download of data packages to all
 * devices. It's fully templated to use one of the structs provided
 * by Alpakaconfig.hpp.
 */
template <typename TAlpaka> struct DeviceData {
    std::size_t id;
  std::size_t numMaps; //! @todo: is this ever used?
    typename TAlpaka::DevHost host;
    typename TAlpaka::DevAcc device;
    typename TAlpaka::Queue queue;
    typename TAlpaka::Event event;
    State state;

    // device maps
    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          Data,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        data;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          Gain,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        gain;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          Pedestal,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        pedestal;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          Mask,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        mask;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          Drift,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        drift;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          GainStage,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        gainStage;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          Value,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        maxValue;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          Photon,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        photon;

    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          PhotonSum,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        sum;

    // host maps
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          Photon,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        photonHost;

    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          PhotonSum,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        sumHost;

    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          Value,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        maxValueHost;

    DeviceData()
        : id(0),
          numMaps(0),
          host(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u)),
          device(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfAcc>(0u)),
          queue(device),
          event(device),
          state(FREE),
          data(alpaka::mem::buf::alloc<Data, typename TAlpaka::Size>(device,
                                                                     0lu)),
          gain(alpaka::mem::buf::alloc<Gain, typename TAlpaka::Size>(device,
                                                                     0lu)),
          pedestal(
              alpaka::mem::buf::alloc<Pedestal, typename TAlpaka::Size>(device,
                                                                        0lu)),
          drift(alpaka::mem::buf::alloc<Drift, typename TAlpaka::Size>(device,
                                                                       0lu)),
          gainStage(
              alpaka::mem::buf::alloc<GainStage, typename TAlpaka::Size>(device,
                                                                         0lu)),
          maxValue(
              alpaka::mem::buf::alloc<Value, typename TAlpaka::Size>(device,
                                                                     0lu)),
          mask(alpaka::mem::buf::alloc<Mask, typename TAlpaka::Size>(device,
                                                                     0lu)),
          manualMask(
              alpaka::mem::buf::alloc<Mask, typename TAlpaka::Size>(device,
                                                                    0lu)),
          photon(alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(device,
                                                                         0lu)),
          sum(alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(device,
                                                                         0lu)),
          driftHost(
              alpaka::mem::buf::alloc<Drift, typename TAlpaka::Size>(host,
                                                                     0lu)),
          gainStageHost(
              alpaka::mem::buf::alloc<GainStage, typename TAlpaka::Size>(host,
                                                                         0lu)),
          maxValueHost(
              alpaka::mem::buf::alloc<Value, typename TAlpaka::Size>(host,
                                                                     0lu)),
          photonHost(
              alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(host,
                                                                      0lu)),
          sumHost(
              alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(host,
                                                                         0lu))
    {
    }
};
