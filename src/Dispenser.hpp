#pragma once

#include "Config.hpp"
#include "Ringbuffer.hpp"

#include "kernel/Correction.hpp"
#include "kernel/Statistics.hpp"
#include "kernel/Summation.hpp"
#include "kernel/Zero.hpp"

#include <iostream>
#include <limits>
#include <mutex>

/**
 * This class manages the upload and download of data packages to all
 * devices. It's fully templated to use one of the structs provided
 * by Alpakaconfig.hpp.
 */
enum State { FREE, PROCESSING, READY };

template <typename TAlpaka> struct DeviceData {
    std::size_t id;
    std::size_t numMaps;
    typename TAlpaka::DevHost host;
    typename TAlpaka::DevAcc device;
    typename TAlpaka::Queue queue;
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
                          Mask,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        mask;
    alpaka::mem::buf::Buf<typename TAlpaka::DevAcc,
                          Mask,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        manualMask;

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
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          Value,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        maxValueHost;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          Drift,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        driftHost;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost,
                          GainStage,
                          typename TAlpaka::Dim,
                          typename TAlpaka::Size>
        gainStageHost;
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

template <typename TAlpaka> class Dispenser {
public:
    /**
     * Dispenser constructor
     * @param Maps-Struct with initial gain
     */
    Dispenser(Maps<Gain, TAlpaka> gainmap, Maps<Mask, TAlpaka> mask);
    /**
     * copy constructor deleted
     */
    Dispenser(const Dispenser& other) = delete;
    /**
     * assign constructor deleted
     */
    Dispenser& operator=(const Dispenser& other) = delete;
    /**
     * Synchronizes all streams with one function call.
     */
    auto synchronize() -> void;
    /**
     * Tries to upload all data packages requiered for the inital offset.
     * Only stops after all data packages are uploaded.
     * @param Maps-Struct with datamaps
     */
    auto uploadPedestaldata(Maps<Data, TAlpaka> data) -> void;
    /**
     * Downloads the pedestal data.
     * @return pedestal pedestal data
     */
    auto downloadPedestaldata() -> Maps<Pedestal, TAlpaka>;
    /**
     * Tries to upload one data package.
     * @param Maps-struct with raw data, offset within the package
     * @return number of frames uploaded from the package
     */
    auto downloadGainStages() -> Maps<GainStage, TAlpaka>;
    /**
     * Tries to upload one data package.
     * @param Maps-struct with raw data, offset within the package
     * @return number of frames uploaded from the package
     */
    auto downloadDriftMaps() -> Maps<Drift, TAlpaka>;
    /**
     * Tries to upload one data package.
     * @param Maps-struct with raw data, offset within the package
     * @return number of frames uploaded from the package
     */
    auto uploadData(Maps<Data, TAlpaka> data, std::size_t offset)
        -> std::size_t;
    /**
     * Tries to download one data package.
     * @param pointer to empty struct for photon and sum maps
     * @return boolean indicating whether maps were downloaded or not
     */
    auto downloadData(Maps<Photon, TAlpaka>* photon,
                      Maps<PhotonSum, TAlpaka>* sum) -> bool;
    /**
     * Returns the a vector with the amount of memory of each device.
     * @return size_array
     */
    auto getMemSize() -> std::vector<std::size_t>;
    /**
     * Returns the a vector with the amount of free memory of each device.
     * @return size_array
     */
    auto getFreeMem() -> std::vector<std::size_t>;

private:
    Maps<Gain, TAlpaka> gain;
    Maps<Mask, TAlpaka> manualMask;
    Maps<Pedestal, TAlpaka> pedestal;
    TAlpaka workdiv;
    bool init;
    Ringbuffer<DeviceData<TAlpaka>*> ringbuffer;
    std::vector<DeviceData<TAlpaka>> devices;
    typename TAlpaka::DevHost host;

    std::mutex mutex;
    std::deque<std::size_t> nextFree;

    /**
     * Initializes all devices. Uploads gain data and creates buffer.
     * @param vector with devices to be initialized
     */
    auto initDevices(std::vector<typename TAlpaka::DevAcc> devs) -> void;
    /**
     * Executes the calibration kernel.
     * @param pointer to raw data and number of frames
     * @return number of frames calculated
     */
    auto calcPedestaldata(Data* data, std::size_t numMaps) -> std::size_t;
    /**
     * Executes summation and correction kernel.
     * @param pointer to raw data and number of frames
     * @return number of frames calculated
     */
    auto calcData(Data* data, std::size_t numMaps) -> std::size_t;
};

template <typename TAlpaka>
Dispenser<TAlpaka>::Dispenser(Maps<Gain, TAlpaka> gainmap,
                              Maps<Mask, TAlpaka> mask)
    : gain(gainmap),
      manualMask(mask),
      workdiv(TAlpaka()),
      init(false),
      ringbuffer(workdiv.STREAMS_PER_DEV *
                 alpaka::pltf::getDevCount<typename TAlpaka::PltfAcc>()),
      host(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u))
{
    std::vector<typename TAlpaka::DevAcc> devs(
        alpaka::pltf::getDevs<typename TAlpaka::PltfAcc>());

    devices.resize(devs.size() * workdiv.STREAMS_PER_DEV);

    initDevices(devs);

    // make room for live pedestal information
    pedestal.numMaps = PEDEMAPS;
    pedestal.data = alpaka::mem::buf::alloc<Pedestal, typename TAlpaka::Size>(
        host, PEDEMAPS * MAPSIZE);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#if (SHOW_DEBUG == false)
    alpaka::mem::buf::pin(pedestal.data);
#endif
#endif
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::initDevices(std::vector<typename TAlpaka::DevAcc> devs)
    -> void
{
    for (std::size_t num = 0; num < devs.size() * workdiv.STREAMS_PER_DEV;
         ++num) {
        // initialize variables
        devices[num].id = num;
        devices[num].device = devs[num / workdiv.STREAMS_PER_DEV];
        devices[num].queue = devs[num / workdiv.STREAMS_PER_DEV];
        devices[num].event = devs[num / workdiv.STREAMS_PER_DEV];
        devices[num].state = FREE;
        // create all buffer on the device
        devices[num].data =
            alpaka::mem::buf::alloc<Data, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
        devices[num].gain =
            alpaka::mem::buf::alloc<Gain, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], GAINMAPS);
        alpaka::mem::view::copy(
            devices[num].queue, devices[num].gain, gain.data, GAINMAPS);
        devices[num].pedestal =
            alpaka::mem::buf::alloc<Pedestal, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], PEDEMAPS);
        devices[num].mask =
            alpaka::mem::buf::alloc<Mask, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
        devices[num].manualMask =
            alpaka::mem::buf::alloc<Mask, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
        devices[num].drift =
            alpaka::mem::buf::alloc<Drift, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], SINGLEMAP);
        devices[num].gainStage =
            alpaka::mem::buf::alloc<GainStage, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], SINGLEMAP);
        devices[num].driftHost =
            alpaka::mem::buf::alloc<Drift, typename TAlpaka::Size>(host,
                                                                   SINGLEMAP);
        devices[num].gainStageHost =
            alpaka::mem::buf::alloc<GainStage, typename TAlpaka::Size>(
                host, SINGLEMAP);
        devices[num].maxValue =
            alpaka::mem::buf::alloc<Value, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], SINGLEMAP);
        devices[num].maxValueHost =
            alpaka::mem::buf::alloc<Value, typename TAlpaka::Size>(host,
                                                                   SINGLEMAP);
        devices[num].photon =
            alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
        devices[num].sum =
            alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES / SUM_FRAMES);
        devices[num].photonHost =
            alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(host,
                                                                    DEV_FRAMES);
        devices[num].sumHost =
            alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(
                host, (DEV_FRAMES / SUM_FRAMES));
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#if (SHOW_DEBUG == false)
        // pin all buffer
        alpaka::mem::buf::pin(devices[num].data);
        alpaka::mem::buf::pin(devices[num].gain);
        alpaka::mem::buf::pin(devices[num].pedestal);
        alpaka::mem::buf::pin(devices[num].mask);
        alpaka::mem::buf::pin(devices[num].drift);
        alpaka::mem::buf::pin(devices[num].gainStage);
        alpaka::mem::buf::pin(devices[num].driftHost);
        alpaka::mem::buf::pin(devices[num].gainStageHost);
        alpaka::mem::buf::pin(devices[num].maxValue);
        alpaka::mem::buf::pin(devices[num].maxValueHost);
        alpaka::mem::buf::pin(devices[num].photon);
        alpaka::mem::buf::pin(devices[num].sum);
        alpaka::mem::buf::pin(devices[num].photonHost);
        alpaka::mem::buf::pin(devices[num].sumHost);
#endif
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
        alpaka::wait::wait(dev.queue);
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::uploadPedestaldata(Maps<Data, TAlpaka> data) -> void
{
    std::size_t offset = 0;
    DEBUG("uploading pedestaldata...");

    // upload all frames cut into smaller packages
    while (offset <= data.numMaps - DEV_FRAMES) {
        offset += calcPedestaldata(
            alpaka::mem::view::getPtrNative(data.data) + offset, DEV_FRAMES);
        DEBUG(offset << "/" << data.numMaps << " pedestalframes uploaded");
    }
    // upload remaining frames
    if (offset != data.numMaps) {
        offset += calcPedestaldata(alpaka::mem::view::getPtrNative(data.data) +
                                       offset,
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
        dev->queue,
        dev->data,
        alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                        Data,
                                        typename TAlpaka::Dim,
                                        typename TAlpaka::Size>(
            data, host, numMaps),
        numMaps);

    // copy offset data from last initialized device
    std::lock_guard<std::mutex> lock(mutex);
    if (nextFree.size() > 0) {
        alpaka::wait::wait(devices[nextFree.back()].queue);
        alpaka::mem::view::copy(dev->queue,
                                dev->pedestal,
                                devices[nextFree.back()].pedestal,
                                PEDEMAPS);
        nextFree.pop_front();
    }
    nextFree.push_back(dev->id);

    if (init == false) {
        ZeroKernel zeroKernel;
        auto const zero(alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
            workdiv.workdiv,
            zeroKernel,
            alpaka::mem::view::getPtrNative(dev->pedestal)));

        alpaka::queue::enqueue(dev->queue, zero);
        alpaka::wait::wait(dev->queue);

        init = true;
    }


    StatisticsKernel StatisticsKernel;
    auto const statistics(alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
        workdiv.workdiv,
        StatisticsKernel,
        alpaka::mem::view::getPtrNative(dev->data),
        dev->numMaps,
        alpaka::mem::view::getPtrNative(dev->pedestal),
        alpaka::mem::view::getPtrNative(dev->mask)));

    alpaka::queue::enqueue(dev->queue, statistics);

    alpaka::wait::wait(dev->queue);
    DEBUG("device " << dev->id << " finished");

    dev->state = FREE;

    if (!ringbuffer.push(dev)) {
        fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
        exit(EXIT_FAILURE);
    }
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    // Maps<Pedestal, Accelerator>
    /*auto ipedestalMaps = downloadPedestaldata();
    uint16_t* iped = new uint16_t[MAPSIZE * PEDEMAPS];
    if (!iped)
        exit(EXIT_FAILURE);

    for (int y = 0; y < DIMY; ++y) {
        for (int x = 0; x < DIMX; ++x) {
            for (int g = 0; g < ipedestalMaps.numMaps; ++g) {
                iped[g * MAPSIZE + y * DIMX + x] =
                    alpaka::mem::view::getPtrNative(
                        ipedestalMaps.data)[g * MAPSIZE + y * DIMX + x]
                        .mean;
            }
        }
    }

    save_image<Photon>(
        "time:" + std::to_string(
            (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":initial_pedestal0:dev" + std::to_string(nextFree.back()),
        iped,
        0);
    save_image<Photon>(
        "time:" + std::to_string(
            (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":initial_pedestal1:dev" + std::to_string(nextFree.back()),
        iped,
        1);
    save_image<Photon>(
        "time:" + std::to_string(
            (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":initial_pedestal2:dev" + std::to_string(nextFree.back()),
        iped,
        2);*/

    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //


    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    // Maps<Pedestal, Accelerator>
    /*
    auto ipedestalMaps = downloadPedestaldata();
    Photon* iped = new Photon[MAPSIZE * PEDEMAPS];
    if (!iped)
        exit(EXIT_FAILURE);

    for (int y = 0; y < DIMY; ++y) {
        for (int x = 0; x < DIMX; ++x) {
            for (int g = 0; g < ipedestalMaps.numMaps; ++g) {
                iped[g * MAPSIZE + y * DIMX + x] =
                    alpaka::mem::view::getPtrNative(
                        ipedestalMaps.data)[g * MAPSIZE + y * DIMX + x]
                        .value;
            }
        }
    }

    save_image<Photon>(
        "time:" +
            std::to_string(
                (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":initial_pedestal0:dev" + std::to_string(nextFree.back()) + ".bmp",
        iped,
        0);
    save_image<Photon>(
        "time:" +
            std::to_string(
                (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":initial_pedestal1:dev" + std::to_string(nextFree.back()) + ".bmp",
        iped,
        1);
    save_image<Photon>(
        "time:" +
            std::to_string(
                (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":initial_pedestal2:dev" + std::to_string(nextFree.back()) + ".bmp",
        iped,
        2);
    */
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //


    return numMaps;
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::downloadPedestaldata() -> Maps<Pedestal, TAlpaka>
{
    DEBUG("downloading pedestaldata...");

    // create handle for the device with the current version of the pedestal
    // maps
    auto current_device = devices[nextFree.back()];

    // get the pedestal data from the device
    alpaka::mem::view::copy(
        current_device.queue,
        pedestal.data,
        current_device.pedestal,
        PEDEMAPS); //! @todo: check this (was PEDEMAPS * MAPSIZE before)

    // wait for copy to finish
    alpaka::wait::wait(current_device.queue, current_device.event);

    return pedestal;
}

//
//
//
//
//
//
//
//
// auto downloadGainStages() -> Maps<GainStage, TAlpaka>;

// auto downloadDriftMaps() -> Maps<Drift, TAlpaka>;
//
//
//
//
//
//
//
//


template <typename TAlpaka>
auto Dispenser<TAlpaka>::uploadData(Maps<Data, TAlpaka> data,
                                    std::size_t offset) -> std::size_t
{
    if (!ringbuffer.isEmpty()) {
        // try uploading one data package
        if (offset <= data.numMaps - DEV_FRAMES) {
            offset +=
                calcData(alpaka::mem::view::getPtrNative(data.data) + offset,
                         DEV_FRAMES);
            DEBUG(offset << "/" << data.numMaps << " frames uploaded");
        }
        // upload remaining frames
        else if (offset != data.numMaps) {
            offset +=
                calcData(alpaka::mem::view::getPtrNative(data.data) + offset,
                         data.numMaps % DEV_FRAMES);
            DEBUG(offset << "/" << data.numMaps << " frames uploaded");
        }
        // force wait for one device to finish since there's no new data
        else {
            alpaka::wait::wait(devices[nextFree.front()].queue,
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

    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    // Maps<Pedestal, Accelerator>
    /*
    auto ipedestalMaps = downloadPedestaldata();
    Photon* iped = new Photon[MAPSIZE * PEDEMAPS];
    if (!iped)
        exit(EXIT_FAILURE);

    for (int y = 0; y < DIMY; ++y) {
        for (int x = 0; x < DIMX; ++x) {
            for (int g = 0; g < ipedestalMaps.numMaps; ++g) {
                iped[g * MAPSIZE + y * DIMX + x] =
                    alpaka::mem::view::getPtrNative(
                        ipedestalMaps.data)[g * MAPSIZE + y * DIMX + x]
                        .value;
            }
        }
    }

    save_image<Photon>(
        "time:" +
            std::to_string(
                (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":running_pedestal0:dev" + std::to_string(nextFree.back()) + ".bmp",
        iped,
        0);
    save_image<Photon>(
        "time:" +
            std::to_string(
                (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":running_pedestal1:dev" + std::to_string(nextFree.back()) + ".bmp",
        iped,
        1);
    save_image<Photon>(
        "time:" +
            std::to_string(
                (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":running_pedestal2:dev" + std::to_string(nextFree.back()) + ".bmp",
        iped,
        2);
    */
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //
    //


    dev->state = PROCESSING;
    dev->numMaps = numMaps;

    alpaka::mem::view::copy(
        dev->queue,
        dev->data,
        alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                        Data,
                                        typename TAlpaka::Dim,
                                        typename TAlpaka::Size>(
            data, host, numMaps),
        numMaps);

    // copy offset data from last device uploaded to
    std::lock_guard<std::mutex> lock(mutex);
    alpaka::wait::wait(dev->queue, devices[nextFree.back()].event);
    DEBUG("device " << devices[nextFree.back()].id << " finished");

    devices[nextFree.back()].state = READY;
    alpaka::mem::view::copy(
        dev->queue, dev->pedestal, devices[nextFree.back()].pedestal, PEDEMAPS);
    nextFree.push_back(dev->id);

    StatisticsKernel statisticsKernel;
    auto const statistics(alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
        workdiv.workdiv,
        statisticsKernel,
        alpaka::mem::view::getPtrNative(dev->data),
        dev->numMaps,
        alpaka::mem::view::getPtrNative(dev->pedestal),
        alpaka::mem::view::getPtrNative(dev->mask)));

    alpaka::queue::enqueue(dev->queue, statistics);
    alpaka::wait::wait(dev->queue);

    CorrectionKernel correctionKernel;
    auto const correction(alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
        workdiv.workdiv,
        correctionKernel,
        alpaka::mem::view::getPtrNative(dev->data),
        alpaka::mem::view::getPtrNative(dev->pedestal),
        alpaka::mem::view::getPtrNative(dev->gain),
        dev->numMaps,
        alpaka::mem::view::getPtrNative(dev->photon),
        alpaka::mem::view::getPtrNative(dev->manualMask),
        alpaka::mem::view::getPtrNative(dev->mask)));

    alpaka::queue::enqueue(dev->queue, correction);
    alpaka::wait::wait(dev->queue);

    SummationKernel summationKernel;
    auto const summation(alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
        workdiv.workdiv,
        summationKernel,
        alpaka::mem::view::getPtrNative(dev->photon),
        SUM_FRAMES,
        dev->numMaps,
        alpaka::mem::view::getPtrNative(dev->sum)));

    alpaka::queue::enqueue(dev->queue, summation);

    save_image<Data>(
        static_cast<std::string>(std::to_string(dev->id) + "data" +
                                 std::to_string(std::rand() % 1000)),
        data,
        DEV_FRAMES - 1);

    // the event is used to wait for pedestal data
    alpaka::queue::enqueue(dev->queue, dev->event);

    return numMaps;
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::downloadData(Maps<Photon, TAlpaka>* photon,
                                      Maps<PhotonSum, TAlpaka>* sum) -> bool
{
    std::lock_guard<std::mutex> lock(mutex);
    struct DeviceData<TAlpaka>* dev = &Dispenser::devices[nextFree.front()];

    // to keep frames in order only download if the longest running device has
    // finished
    if (dev->state != READY)
        return false;

    photon->numMaps = dev->numMaps;
    alpaka::mem::view::copy(
        dev->queue, dev->photonHost, dev->photon, dev->numMaps);
    photon->data = dev->photonHost;
    photon->header = true;

    sum->numMaps = dev->numMaps / SUM_FRAMES;
    alpaka::mem::view::copy(
        dev->queue, dev->sumHost, dev->sum, (dev->numMaps / SUM_FRAMES));
    sum->data = dev->sumHost;
    sum->header = true;

    alpaka::wait::wait(dev->queue, dev->event);

    dev->state = FREE;
    nextFree.pop_front();
    ringbuffer.push(dev);
    DEBUG("device " << dev->id << " freed");


    save_image<Photon>(
        static_cast<std::string>(std::to_string(dev->id) + "First" +
                                 std::to_string(std::rand() % 1000)),
        alpaka::mem::view::getPtrNative(photon->data),
        0);
    save_image<Photon>(
        static_cast<std::string>(std::to_string(dev->id) + "Last" +
                                 std::to_string(std::rand() % 1000)),
        alpaka::mem::view::getPtrNative(photon->data),
        DEV_FRAMES - 1);


    return true;
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::getMemSize() -> std::vector<std::size_t>
{
    std::vector<std::size_t> sizes(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
        sizes[i] = alpaka::dev::getMemBytes(devices[i].device);
    }

    return sizes;
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::getFreeMem() -> std::vector<std::size_t>
{
    std::vector<std::size_t> sizes(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
        sizes[i] = alpaka::dev::getFreeMemBytes(devices[i].device);
    }

    return sizes;
}
