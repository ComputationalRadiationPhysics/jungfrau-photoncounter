#pragma once

#include "Config.hpp"
#include "Ringbuffer.hpp"

#include "kernel/Calibration.hpp"
#include "kernel/Correction.hpp"
#include "kernel/Summation.hpp"


enum State {FREE, PROCESSING, READY};

template <TAlpaka> struct DeviceData {
    TAlpaka::DevAcc device;
    TAlpaka::Stream stream;
    State state;
    Pedestal* pedestal;
    Photon* photon;
    PhotonSum* sum;
};

template <TAlpaka> class Dispenser {
public:
    Dispenser(Maps<Gain> gainmap);
    Dispenser(const Dispenser& other) = delete;
    Dispenser& operator=(const Dispenser& other) = delete;
    ~Dispenser();

private:
    Maps<Gain> gain; 
    Ringbuffer<DeviceData*> ringbuffer;
    std::vector<DeviceData> devices;

    void initDevices(std::vector<TAlpka::PltfAcc> devs);
};

template <TAlpaka>
Dispenser<TAlpaka>::Dispenser(Maps<Gain> gainmap) : gain(gainmap)
{
    std::vector<TAlpka::PltfAcc> devs(alpaka::pltf::getDevs<PltfAcc>()); 

    ringbuffer(Ringbuffer(STREAMS_PER_GPU * devs.size()));
    devices.resize(devs.getSize());

    initDevices(devs);

}

template <TAlpaka> Dispenser<TAlpaka>::~Dispenser() {}

template <TAlpaka>
Dispenser<TAlpaka>::initDevices(std::vector<TAlpaka::PltfAcc> devs)
{
    for (size_t num = 0; num < devs.size(), ++num) {
        devices[num].DevAcc(devs[num]);
        devices[num].stream(TAlpaka::DevAcc);
        devices[num].state(Free);
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


