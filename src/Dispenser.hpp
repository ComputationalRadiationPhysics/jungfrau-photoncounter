#pragma once

#include "Config.hpp"
#include "Ringbuffer.hpp"

#include "kernel/Calibration.hpp"
#include "kernel/Correction.hpp"
#include "kernel/Summation.hpp"


enum State {FREE, PROCESSING, READY};

template <TAlpaka> struct DeviceData {
    std::size_t id;
    TAlpaka::DevAcc device;
    TAlpaka::Stream stream;
    State state;
    Data* data;
    Gain* gain;
    Pedestal* pedestal;
    Photon* photon;
    PhotonSum* sum;
};

template <TAlpaka> class Dispenser {
public:
    Dispenser(Maps<Gain> gainmap, TAlpaka workdivSruct);
    Dispenser(const Dispenser& other) = delete;
    Dispenser& operator=(const Dispenser& other) = delete;
    ~Dispenser();

    auto synchronize() -> void;
    auto uploadPedestaldata(Maps<Data>* data) -> void;
    auto uploadData(Maps<Data>* data) -> void;
    auto downloadData() -> Maps<Data>*;

private:
    Maps<Gain>* gain; 
    Ringbuffer<DeviceData*> ringbuffer;
    std::vector<DeviceData> devices;
    TAlpaka workdiv;

    auto initDevices(std::vector<TAlpka::PltfAcc> devs) -> void;
    auto calcPedestaldata(Data* data, std::size_t numMaps) -> std::size_t;
    auto calcData(Data* data, std::size_t numMaps) -> std::size_t;
};

template <TAlpaka>
Dispenser<TAlpaka>::Dispenser(Maps<Gain> gainmap, TAlpaka workdivStruct)
    : gain(gainmap) : workdiv(workdivSruct)
{
    std::vector<TAlpka::PltfAcc> devs(alpaka::pltf::getDevs<PltfAcc>()); 

    ringbuffer(Ringbuffer(STREAMS_PER_GPU * devs.size()));
    devices.resize(devs.getSize());

    initDevices(devs);
}

template <TAlpaka> Dispenser<TAlpaka>::~Dispenser() {}

template <TAlpaka>
auto Dispenser<TAlpaka>::initDevices(std::vector<TAlpaka::PltfAcc> devs) -> void
{
    for (size_t num = 0; num < devs.size() * STREAMS_PER_GPU, ++num) {
        devices[num].id = num;
        devices[num].device(devs[num / STREAMS_PER_GPU]);
        devices[num].stream(TAlpaka::DevAcc);
        devices[num].state(FREE);
        devices[num].data(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<Data, TAlpka::Size>(
                devices[num].DevAcc, GPU_FRAMES * MAPSIZE)));
        devices[num].gain(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<Gain, TAlpka::Size>(
                devices[num].DevAcc, GAINMAPS * MAPSIZE)));
        alpaka::mem::view::copy(
            device[num].stream, gain, device[num].gain, MAPSIZE * GAINMAPS);
        devices[num].pedestal(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<Pedestal, TAlpka::Size>(
                devices[num].DevAcc, PEDEMAPS * MAPSIZE)));
        devices[num].photon(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<Photon, TAlpka::Size>(
                devices[num].DevAcc, GPU_FRAMES * MAPSIZE)));
        devices[num].sum(alpaka::mem::view::getPtrNative(
            alpaka::mem::buf::alloc<PhotonSum, TAlpka::Size>(
                devices[num].DevAcc, (GPU_FRAMES / SUM_FRAMES) * MAPSIZE)));
    }
}

template <TAlpaka> auto Dispenser<TAlpaka>::synchroize() -> void
{
    for (struct DeviceData dev : devices)
        alpaka::wait::wait(dev.stream);
}

template <TAlpaka> auto Dispenser<TAlpaka>::uploadPedestal(Maps<Data>* data) -> void
{
    std::size_t offset = 0;

    while (offset <= data->numMaps - GPU_FRAMES) {
        offset += calcPedestaldata(data->dataPointer + offset, GPU_FRAMES);
    }
    while (offset /= data.numMaps) {
        offset += calcPedestaldata(data->dataPointer + offset,
                                   data->numMaps - GPU_FRAMES);
    }
}

template <TAlpaka>
auto Dispenser<TAlpaka>::calcPedestaldata(Data* data, std::size_t numMaps)
    ->std::size_t
{
    struct deviceData* dev;    
    if (!ringbuffer.pop(dev))
        return 0;

    dev->state = PROCESSING;

    alpaka::mem::view::copy(
        TAlpaka::Stream, data, dev->data, numMaps * (MAPSIZE + OFFSET));
    alpaka::mem::view::copy(dev->stream,
                            devices[(dev->id - 1) % devices.size()].pedestal,
                            ddev->pedestal,
                            GAINMAPS * MAPSIZE);

    CalibrationKernel calibrationKernel;
    
    auto const calibration(
        alpaka::exec::create<Acc>(workdiv.workdiv,
                                  calibrationKernel,
                                  data,
                                  dev->pedestal,
                                  GPU_FRAMES));

    alpaka::stream::enqueue(dev->stream, calibration);
    alpaka::wait::wait(dev->stream);

    dev->state = FREE;

    if(!ringbuffer.push(dev)) {
        fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
        exit(EXIT_FAILURE);
    }

    return GPU_FRAMES;
}

template <TAlpaka>
auto Dispenser<TAlpaka>::uploadData(Maps<Data>* data) -> void {
    std::size_t offset = 0;

    while (offset <= data->numMaps - GPU_FRAMES) {
        offset += calcData(data->dataPointer + offset, GPU_FRAMES);
    }
    while (offset /= data.numMaps) {
        offset += calcData(data->dataPointer + offset,
                                   data->numMaps - GPU_FRAMES);
    }
}


template <TAlpaka>
auto Dispenser<TAlpaka>::calcData(Data* data, std::size_t numMaps)
    -> std::size_t
{
    struct deviceData* dev;    
    if (!ringbuffer.pop(dev))
        return 0;

    dev->state = PROCESSING;

    alpaka::mem::view::copy(
        TAlpaka::Stream, data, dev->data, numMaps * (MAPSIZE + OFFSET));
    alpaka::mem::view::copy(dev->stream,
                            devices[(dev->id - 1) % devices.size()].pedestal,
                            ddev->pedestal,
                            GAINMAPS * MAPSIZE);

    CorrectionKernel correctionKernel;
    SummationKernel summationKernel;

    auto const correction(
        alpaka::exec::create<Acc>(workdiv.workdiv,
                                  correctionKernel,
                                  data,
                                  dev->pedestal,
                                  dev->gain,
                                  GPU_FRAMES,
                                  alpaka::mem::view::getPtrNative(photon_c)));

///////////////////////////////////////////////////////////
    auto const summation(
        alpaka::exec::create<Acc>(workdiv,
                                  summationKernel,
                                  alpaka::mem::view::getPtrNative(photon_c),
                                  SUM_FRAMES,
                                  1000u,
                                  alpaka::mem::view::getPtrNative(sum_c)));

    alpaka::stream::enqueue(stream, correction);
    alpaka::wait::wait(stream);

    alpaka::stream::enqueue(stream, summation);
    alpaka::wait::wait(stream);

    return GPU_FRAMES;   
}

template <TAlpaka>
auto Dispenser<TAlpaka>::downloadData() -> Maps<Data>*;



















