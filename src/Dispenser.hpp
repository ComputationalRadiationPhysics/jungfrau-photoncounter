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
    typename TAlpaka::DevAcc device;
    typename TAlpaka::Stream stream;
    typename TAlpaka::Event event;
    State state;
    Data* data;
    Gain* gain;
    Pedestal* pedestal;
    Photon* photon;
    PhotonSum* sum;
};

template <typename TAlpaka> class Dispenser {
public:
    Dispenser(Maps<Gain> gainmap, TAlpaka workdivSruct);
    Dispenser(const Dispenser& other) = delete;
    Dispenser& operator=(const Dispenser& other) = delete;
    ~Dispenser();

    auto synchronize() -> void;
    auto uploadPedestaldata(Maps<Data>* data) -> void;
    auto uploadData(Maps<Data>* data) -> void;
    auto downloadData(Maps<Photon>* photon, Maps<PhotonSum>* sum) -> void;

private:
    Maps<Gain>* gain;
    Ringbuffer<DeviceData<TAlpaka>*> ringbuffer;
    std::vector<DeviceData<TAlpaka>> devices;
    TAlpaka workdiv;

    std::mutex mutex;
    std::size_t nextFree;

    auto initDevices(std::vector<typename TAlpaka::PltfAcc> devs) -> void;
    auto calcPedestaldata(Data* data, std::size_t numMaps) -> std::size_t;
    auto calcData(Data* data, std::size_t numMaps) -> std::size_t;
};

template <typename TAlpaka>
Dispenser<TAlpaka>::Dispenser(Maps<Gain> gainmap, TAlpaka workdivStruct)
    : gain(gainmap), workdiv(workdivStruct), nextFree(0)
{
    std::vector<typename TAlpaka::PltfAcc> devs(
        alpaka::pltf::getDevs<typename TAlpaka::PltfAcc>());

    ringbuffer(Ringbuffer<DeviceData<TAlpaka>*>(STREAMS_PER_GPU * devs.size()));
    devices.resize(devs.getSize());

    initDevices(devs);
}

template <typename TAlpaka> Dispenser<TAlpaka>::~Dispenser() {}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::initDevices(
    std::vector<typename TAlpaka::PltfAcc> devs) -> void
{
    for (size_t num = 0; num < devs.size() * STREAMS_PER_GPU; ++num) {
        devices[num].id = num;
        devices[num].device(devs[num / STREAMS_PER_GPU]);
        devices[num].stream(TAlpaka::DevAcc);
        devices[num].state(FREE);
        devices[num].data(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<Data, typename TAlpaka::Size>(
                devices[num].DevAcc, GPU_FRAMES * MAPSIZE)));
        devices[num].gain(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<Gain, typename TAlpaka::Size>(
                devices[num].DevAcc, GAINMAPS * MAPSIZE)));
        alpaka::mem::view::copy(
            devices[num].stream, gain, devices[num].gain, MAPSIZE * GAINMAPS);
        devices[num].pedestal(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<Pedestal, typename TAlpaka::Size>(
                devices[num].DevAcc, PEDEMAPS * MAPSIZE)));
        devices[num].photon(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
                devices[num].DevAcc, GPU_FRAMES * MAPSIZE)));
        devices[num].sum(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(
                devices[num].DevAcc, (GPU_FRAMES / SUM_FRAMES) * MAPSIZE)));
    }
}

template <typename TAlpaka> auto Dispenser<TAlpaka>::synchronize() -> void
{
    for (struct DeviceData<TAlpaka> dev : devices)
        alpaka::wait::wait(dev.stream);
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::uploadPedestaldata(Maps<Data>* data) -> void
{
    std::size_t offset = 0;

    while (offset <= data->numMaps - GPU_FRAMES) {
        offset += calcPedestaldata(data->dataPointer + offset, GPU_FRAMES);
    }
    while (offset /= data->numMaps) {
        offset += calcPedestaldata(data->dataPointer + offset,
                                   data->numMaps - GPU_FRAMES);
    }
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::calcPedestaldata(Data* data, std::size_t numMaps)
    -> std::size_t
{
    struct deviceData* dev;
    if (!ringbuffer.pop(dev))
        return 0;

    dev->state = PROCESSING;
    dev->numMaps = numMaps;

    alpaka::mem::view::copy(
        TAlpaka::Stream, data, dev->data, numMaps * (MAPSIZE + FRAMEOFFSET));
    alpaka::mem::view::copy(dev->stream,
                            devices[(dev->id - 1) % devices.size()].pedestal,
                            dev->pedestal,
                            GAINMAPS * MAPSIZE);

    CalibrationKernel calibrationKernel;

    auto const calibration(alpaka::exec::create<typename TAlpaka::Acc>(
        workdiv.workdiv, calibrationKernel, data, dev->pedestal, GPU_FRAMES));

    alpaka::stream::enqueue(dev->stream, calibration);
    alpaka::wait::wait(dev->stream);

    dev->state = FREE;

    if (!ringbuffer.push(dev)) {
        fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
        exit(EXIT_FAILURE);
    }

    return GPU_FRAMES;
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::uploadData(Maps<Data>* data) -> void
{
    std::size_t offset = 0;

    while (offset <= data->numMaps - GPU_FRAMES) {
        offset += calcData(data->dataPointer + offset, GPU_FRAMES);
    }
    while (offset /= data->numMaps) {
        offset +=
            calcData(data->dataPointer + offset, data->numMaps - GPU_FRAMES);
    }
}


template <typename TAlpaka>
auto Dispenser<TAlpaka>::calcData(Data* data, std::size_t numMaps)
    -> std::size_t
{
    struct deviceData* dev;
    if (!ringbuffer.pop(dev))
        return 0;

    dev->state = PROCESSING;
    dev->numMaps = numMaps;

    alpaka::mem::view::copy(
        TAlpaka::Stream, data, dev->data, numMaps * (MAPSIZE + FRAMEOFFSET));
    alpaka::wait::wait(dev->stream,
                       devices[(dev->id - 1) % ringbuffer->getSize()]->event);
    devices[(dev->id - 1) % ringbuffer->getSize()]->state = READY;
    alpaka::mem::view::copy(dev->stream,
                            devices[(dev->id - 1) % devices.size()].pedestal,
                            dev->pedestal,
                            GAINMAPS * MAPSIZE);

    CorrectionKernel correctionKernel;
    auto const correction(alpaka::exec::create<typename TAlpaka::Acc>(
        workdiv.workdiv,
        correctionKernel,
        data,
        dev->pedestal,
        dev->gain,
        GPU_FRAMES,
        alpaka::mem::view::getPtrNative(dev->photon_c)));

    alpaka::stream::enqueue(dev->stream, correction);
    alpaka::wait::wait(dev->stream);

    SummationKernel summationKernel;
    auto const summation(alpaka::exec::create<typename TAlpaka::Acc>(
        workdiv.workdiv,
        summationKernel,
        alpaka::mem::view::getPtrNative(dev->photon_c),
        SUM_FRAMES,
        1000u,
        alpaka::mem::view::getPtrNative(dev->sum_c)));

    alpaka::stream::enqueue(dev->stream, summation);
    alpaka::wait::wait(dev->stream);

    dev->event(alpaka::event::Event<typename TAlpaka::Stream>(dev->device));
    alpaka::stream::enqueue(dev->stream, dev->event);

    return GPU_FRAMES;
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::downloadData(Maps<Photon>* photon, Maps<PhotonSum>* sum)
    -> void
{
    std::lock_guard<std::mutex> lock(mutex); 
    struct DeviceData<TAlpaka>* dev = &Dispenser::devices[nextFree];

    if (dev->state != READY)
        return;

    photon->dataPointer = alpaka::mem::view::getPtrNative(
        alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
            TAlpaka::DevHost, dev->numFrames * MAPSIZE));

    sum->dataPointer = alpaka::mem::view::getPtrNative(
        alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
            TAlpaka::DevHost, dev->numMaps * MAPSIZE));

    photon->numMaps = dev->numMaps;
    alpaka::mem::view::copy(dev->stream,
                            photon->dataPointer,
                            dev->photon_c,
                            dev->numMaps * MAPSIZE);
    photon->header = true;
    
    sum->numMaps = dev->numMaps / SUM_FRAMES;
    alpaka::mem::view::copy(dev->stream,
                            sum->dataPointer,
                            dev->sum_c,
                            dev->numMaps * MAPSIZE);
    sum->header = true;

    dev->state = FREE;
    nextFree = (nextFree+1) % ringbuffer.getSize();
}
