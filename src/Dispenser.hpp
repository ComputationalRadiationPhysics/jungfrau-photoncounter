#pragma once

#include "Config.hpp"
#include "Ringbuffer.hpp"

#include "kernel/Calibration.hpp"
#include "kernel/Correction.hpp"
#include "kernel/Summation.hpp"

#include <mutex>


enum State { FREE, PROCESSING, READY };

template <typename TAlpaka> struct DeviceData {
    std::size_t id;
    std::size_t numMaps;
    typename TAlpaka::DevHost host;
    typename TAlpaka::DevAcc device;
    typename TAlpaka::Stream stream;
    typename TAlpaka::Event event;
    State state;
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
                          Photon,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        photon;
    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          PhotonSum,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        sum;
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


    DeviceData()
        : id(0),
          numMaps(0),
          host(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u)),
          device(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfAcc>(0u)),
          stream(device),
          event(device),
          state(FREE),
          data(alpaka::mem::buf::alloc<Data, typename TAlpaka::Size>(device,
                                                                     0lu)),
          gain(alpaka::mem::buf::alloc<Gain, typename TAlpaka::Size>(device,
                                                                     0lu)),
          pedestal(
              alpaka::mem::buf::alloc<Pedestal, typename TAlpaka::Size>(device,
                                                                        0lu)),
          photon(alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(device,
                                                                         0lu)),
          sum(alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(device,
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

template <typename TAlpaka> class Dispenser {
public:
    Dispenser(Maps<Gain, TAlpaka> gainmap, TAlpaka workdivSruct);
    Dispenser(const Dispenser& other) = delete;
    Dispenser& operator=(const Dispenser& other) = delete;
    ~Dispenser();

    auto synchronize() -> void;
    auto uploadPedestaldata(Maps<Data, TAlpaka> data) -> void;
    auto uploadData(Maps<Data, TAlpaka> data, std::size_t offset)
        -> std::size_t;
    auto downloadData(Maps<Photon, TAlpaka>* photon,
                      Maps<PhotonSum, TAlpaka>* sum) -> bool;

private:
    Maps<Gain, TAlpaka> gain;
    Ringbuffer<DeviceData<TAlpaka>*> ringbuffer;
    std::vector<DeviceData<TAlpaka>> devices;
    typename TAlpaka::DevHost host;
    TAlpaka workdiv;

    std::mutex mutex;
    std::deque<std::size_t> nextFree;

    auto initDevices(std::vector<typename TAlpaka::DevAcc> devs) -> void;
    auto calcPedestaldata(Data* data, std::size_t numMaps) -> std::size_t;
    auto calcData(Data* data, std::size_t numMaps) -> std::size_t;
};

template <typename TAlpaka>
Dispenser<TAlpaka>::Dispenser(Maps<Gain, TAlpaka> gainmap,
                              TAlpaka workdivStruct)
    : gain(gainmap),
      ringbuffer(STREAMS_PER_DEV *
                 alpaka::pltf::getDevCount<typename TAlpaka::PltfAcc>()),
      host(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u)),
      workdiv(workdivStruct)
{
    std::vector<typename TAlpaka::DevAcc> devs(
        alpaka::pltf::getDevs<typename TAlpaka::PltfAcc>());

    devices.resize(devs.size() * STREAMS_PER_DEV);

    initDevices(devs);
}

template <typename TAlpaka> Dispenser<TAlpaka>::~Dispenser() {}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::initDevices(std::vector<typename TAlpaka::DevAcc> devs)
    -> void
{
    for (size_t num = 0; num < devs.size() * STREAMS_PER_DEV; ++num) {
        devices[num].id = num;
        devices[num].device = devs[num / STREAMS_PER_DEV];
        devices[num].stream = devs[num / STREAMS_PER_DEV];
        devices[num].event = devs[num / STREAMS_PER_DEV];
        devices[num].state = FREE;
        devices[num].data =
            alpaka::mem::buf::alloc<Data, typename TAlpaka::Size>(
                devs[num / STREAMS_PER_DEV],
                DEV_FRAMES * (MAPSIZE + FRAMEOFFSET));
        devices[num].gain =
            alpaka::mem::buf::alloc<Gain, typename TAlpaka::Size>(
                devs[num / STREAMS_PER_DEV], GAINMAPS * MAPSIZE);
        alpaka::mem::view::copy(devices[num].stream,
                                devices[num].gain,
                                gain.data,
                                MAPSIZE * GAINMAPS);
        devices[num].pedestal =
            alpaka::mem::buf::alloc<Pedestal, typename TAlpaka::Size>(
                devs[num / STREAMS_PER_DEV], PEDEMAPS * MAPSIZE);
        devices[num].photon =
            alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
                devs[num / STREAMS_PER_DEV],
                DEV_FRAMES * (MAPSIZE + FRAMEOFFSET));
        devices[num].sum =
            alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(
                devs[num / STREAMS_PER_DEV],
                (DEV_FRAMES / SUM_FRAMES) * MAPSIZE);
        devices[num].photonHost =
            alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
                host, DEV_FRAMES * (MAPSIZE + FRAMEOFFSET));
        devices[num].sumHost =
            alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(
                host, (DEV_FRAMES / SUM_FRAMES) * MAPSIZE);
#if (SHOW_DEBUG == false)
        alpaka::mem::buf::pin(devices[num].data);
        alpaka::mem::buf::pin(devices[num].gain);
        alpaka::mem::buf::pin(devices[num].pedestal);
        alpaka::mem::buf::pin(devices[num].photon);
        alpaka::mem::buf::pin(devices[num].sum);
        alpaka::mem::buf::pin(devices[num].photonHost);
        alpaka::mem::buf::pin(devices[num].sumHost);
#endif


        if (!ringbuffer.push(&devices[num])) {
            fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
            exit(EXIT_FAILURE);
        }
        DEBUG("Device # " << (num + 1) << " init");
    }
}

template <typename TAlpaka> auto Dispenser<TAlpaka>::synchronize() -> void
{
    for (struct DeviceData<TAlpaka> dev : devices)
        alpaka::wait::wait(dev.stream);
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::uploadPedestaldata(Maps<Data, TAlpaka> data) -> void
{
    std::size_t offset = 0;
    DEBUG("uploading pedestaldata...");

    while (offset <= data.numMaps - DEV_FRAMES) {
        offset += calcPedestaldata(alpaka::mem::view::getPtrNative(data.data) +
                                       (offset * (MAPSIZE + FRAMEOFFSET)),
                                   DEV_FRAMES);
        DEBUG(offset << "/" << data.numMaps << " pedestalframes uploaded");
    }
    if (offset != data.numMaps) {
        offset += calcPedestaldata(alpaka::mem::view::getPtrNative(data.data) +
                                       (offset * (MAPSIZE + FRAMEOFFSET)),
                                   data.numMaps % DEV_FRAMES);
        DEBUG(offset << "/" << data.numMaps << " pedestalframes uploaded");
    }
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::calcPedestaldata(Data* data, std::size_t numMaps)
    -> std::size_t
{
    DeviceData<TAlpaka>* dev;
    if (!ringbuffer.pop(dev))
        return 0;

    dev->state = PROCESSING;
    dev->numMaps = numMaps;

    alpaka::mem::view::copy(
        dev->stream,
        dev->data,
        alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                        Data,
                                        typename TAlpaka::Dim,
                                        typename TAlpaka::Size>(
            data, host, (numMaps * (MAPSIZE + FRAMEOFFSET))),
        numMaps * (MAPSIZE + FRAMEOFFSET));

    std::lock_guard<std::mutex> lock(mutex);
    if (nextFree.size() > 0) {
        alpaka::wait::wait(devices[nextFree.back()].stream);
        alpaka::mem::view::copy(dev->stream,
                                dev->pedestal,
                                devices[nextFree.back()].pedestal,
                                PEDEMAPS * MAPSIZE);
        nextFree.pop_front();
    }
    nextFree.push_back(dev->id);

    CalibrationKernel calibrationKernel;

    auto const calibration(alpaka::exec::create<typename TAlpaka::Acc>(
        workdiv.workdiv,
        calibrationKernel,
        alpaka::mem::view::getPtrNative(dev->data),
        alpaka::mem::view::getPtrNative(dev->pedestal),
        dev->numMaps));

    alpaka::stream::enqueue(dev->stream, calibration);
    alpaka::wait::wait(dev->stream);
    DEBUG("device " << dev->id << " finished");

    dev->state = FREE;

    if (!ringbuffer.push(dev)) {
        fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
        exit(EXIT_FAILURE);
    }

    return numMaps;
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::uploadData(Maps<Data, TAlpaka> data,
                                    std::size_t offset) -> std::size_t
{
    if (!ringbuffer.isEmpty()) {
        if (offset <= data.numMaps - DEV_FRAMES) {
            offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                   (offset * (MAPSIZE + FRAMEOFFSET)),
                               DEV_FRAMES);
            DEBUG(offset << "/" << data.numMaps << " frames uploaded");
        }
        else if (offset != data.numMaps) {
            offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                   (offset * (MAPSIZE + FRAMEOFFSET)),
                               data.numMaps % DEV_FRAMES);
            DEBUG(offset << "/" << data.numMaps << " frames uploaded");
        }
        else {
            alpaka::wait::wait(devices[nextFree.front()].stream,
                               devices[nextFree.front()].event);
            devices[nextFree.front()].state = READY;
        }
    }

    return offset;
}


template <typename TAlpaka>
auto Dispenser<TAlpaka>::calcData(Data* data, std::size_t numMaps)
    -> std::size_t
{
    DeviceData<TAlpaka>* dev;
    if (!ringbuffer.pop(dev))
        return 0;

    dev->state = PROCESSING;
    dev->numMaps = numMaps;

    alpaka::mem::view::copy(
        dev->stream,
        dev->data,
        alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                        Data,
                                        typename TAlpaka::Dim,
                                        typename TAlpaka::Size>(
            data, host, (numMaps * (MAPSIZE + FRAMEOFFSET))),
        numMaps * (MAPSIZE + FRAMEOFFSET));

    std::lock_guard<std::mutex> lock(mutex);
    alpaka::wait::wait(dev->stream, devices[nextFree.back()].event);
    DEBUG("device " << devices[nextFree.back()].id << " finished");

    devices[nextFree.back()].state = READY;
    alpaka::mem::view::copy(dev->stream,
                            dev->pedestal,
                            devices[nextFree.back()].pedestal,
                            PEDEMAPS * MAPSIZE);
    nextFree.push_back(dev->id);

    CorrectionKernel correctionKernel;
    auto const correction(alpaka::exec::create<typename TAlpaka::Acc>(
        workdiv.workdiv,
        correctionKernel,
        alpaka::mem::view::getPtrNative(dev->data),
        alpaka::mem::view::getPtrNative(dev->pedestal),
        alpaka::mem::view::getPtrNative(dev->gain),
        dev->numMaps,
        alpaka::mem::view::getPtrNative(dev->photon)));

    alpaka::stream::enqueue(dev->stream, correction);
    alpaka::wait::wait(dev->stream);

    SummationKernel summationKernel;
    auto const summation(alpaka::exec::create<typename TAlpaka::Acc>(
        workdiv.workdiv,
        summationKernel,
        alpaka::mem::view::getPtrNative(dev->photon),
        SUM_FRAMES,
        dev->numMaps,
        alpaka::mem::view::getPtrNative(dev->sum)));

    alpaka::stream::enqueue(dev->stream, summation);

    save_image<Data>(
        static_cast<std::string>(std::to_string(dev->id) + "data.bmp"),
        data,
        DEV_FRAMES - 1);

    alpaka::stream::enqueue(dev->stream, dev->event);

    return numMaps;
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::downloadData(Maps<Photon, TAlpaka>* photon,
                                      Maps<PhotonSum, TAlpaka>* sum) -> bool
{
    std::lock_guard<std::mutex> lock(mutex);
    struct DeviceData<TAlpaka>* dev = &Dispenser::devices[nextFree.front()];

    if (dev->state != READY)
        return false;

    photon->numMaps = dev->numMaps;
    alpaka::mem::view::copy(dev->stream,
                            dev->photonHost,
                            dev->photon,
                            dev->numMaps * (MAPSIZE + FRAMEOFFSET));
    photon->data = dev->photonHost;
    photon->header = true;

    sum->numMaps = dev->numMaps / SUM_FRAMES;
    alpaka::mem::view::copy(dev->stream,
                            dev->sumHost,
                            dev->sum,
                            (dev->numMaps / SUM_FRAMES) * MAPSIZE);
    sum->data = dev->sumHost;
    sum->header = true;

    dev->state = FREE;
    nextFree.pop_front();
    ringbuffer.push(dev);
    DEBUG("device " << dev->id << " freed");


    save_image<Photon>(
        static_cast<std::string>(std::to_string(dev->id) + "First.bmp"),
        alpaka::mem::view::getPtrNative(photon->data),
        0);
    save_image<Photon>(
        static_cast<std::string>(std::to_string(dev->id) + "Last.bmp"),
        alpaka::mem::view::getPtrNative(photon->data),
        DEV_FRAMES - 1);


    return true;
}
