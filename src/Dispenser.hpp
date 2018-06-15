#pragma once

#include "Config.hpp"
#include "Ringbuffer.hpp"

#include "kernel/Calibration.hpp"
#include "kernel/Correction.hpp"
#include "kernel/Summation.hpp"

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
    /**
     * Dispenser constructor
     * @param Maps-Struct with initial gain
     */
    Dispenser(Maps<Gain, TAlpaka> gainmap);
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
    Maps<Pedestal, TAlpaka> pedestal;
    TAlpaka workdiv;
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
Dispenser<TAlpaka>::Dispenser(Maps<Gain, TAlpaka> gainmap)
    : gain(gainmap),
      workdiv(TAlpaka()),
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
        devices[num].stream = devs[num / workdiv.STREAMS_PER_DEV];
        devices[num].event = devs[num / workdiv.STREAMS_PER_DEV];
        devices[num].state = FREE;
        // create all buffer on the device
        devices[num].data =
            alpaka::mem::buf::alloc<Data, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV],
                DEV_FRAMES * (MAPSIZE + FRAMEOFFSET));
        devices[num].gain =
            alpaka::mem::buf::alloc<Gain, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], GAINMAPS * MAPSIZE);
        alpaka::mem::view::copy(devices[num].stream,
                                devices[num].gain,
                                gain.data,
                                MAPSIZE * GAINMAPS);
        devices[num].pedestal =
            alpaka::mem::buf::alloc<Pedestal, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV], PEDEMAPS * MAPSIZE);
        devices[num].photon =
            alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV],
                DEV_FRAMES * (MAPSIZE + FRAMEOFFSET));
        devices[num].sum =
            alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(
                devs[num / workdiv.STREAMS_PER_DEV],
                (DEV_FRAMES / SUM_FRAMES) * MAPSIZE);
        devices[num].photonHost =
            alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
                host, DEV_FRAMES * (MAPSIZE + FRAMEOFFSET));
        devices[num].sumHost =
            alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(
                host, (DEV_FRAMES / SUM_FRAMES) * MAPSIZE);
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#if (SHOW_DEBUG == false)
        // pin all buffer
        alpaka::mem::buf::pin(devices[num].data);
        alpaka::mem::buf::pin(devices[num].gain);
        alpaka::mem::buf::pin(devices[num].pedestal);
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
        alpaka::wait::wait(dev.stream);
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::uploadPedestaldata(Maps<Data, TAlpaka> data) -> void
{
    std::size_t offset = 0;
    DEBUG("uploading pedestaldata...");

    // upload all frames cut into smaller packages
    while (offset <= data.numMaps - DEV_FRAMES) {
        offset += calcPedestaldata(alpaka::mem::view::getPtrNative(data.data) +
                                       (offset * (MAPSIZE + FRAMEOFFSET)),
                                   DEV_FRAMES);
        DEBUG(offset << "/" << data.numMaps << " pedestalframes uploaded");
    }
    // upload remaining frames
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

    // copy offset data from last initialized device
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
        "time:" + std::to_string(
            (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":initial_pedestal0:dev" + std::to_string(nextFree.back()) +
            ".bmp",
        iped,
        0);
    save_image<Photon>(
        "time:" + std::to_string(
            (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":initial_pedestal1:dev" + std::to_string(nextFree.back()) +
            ".bmp",
        iped,
        1);
    save_image<Photon>(
        "time:" + std::to_string(
            (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":initial_pedestal2:dev" + std::to_string(nextFree.back()) +
            ".bmp",
        iped,
        2);

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
    alpaka::mem::view::copy(current_device.stream,
                            pedestal.data,
                            current_device.pedestal,
                            PEDEMAPS * MAPSIZE);

    // wait for copy to finish
    alpaka::wait::wait(current_device.stream, current_device.event);

    return pedestal;
}

template <typename TAlpaka>
auto Dispenser<TAlpaka>::uploadData(Maps<Data, TAlpaka> data,
                                    std::size_t offset) -> std::size_t
{
    if (!ringbuffer.isEmpty()) {
        // try uploading one data package
        if (offset <= data.numMaps - DEV_FRAMES) {
            offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                   (offset * (MAPSIZE + FRAMEOFFSET)),
                               DEV_FRAMES);
            DEBUG(offset << "/" << data.numMaps << " frames uploaded");
        }
        // upload remaining frames
        else if (offset != data.numMaps) {
            offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                   (offset * (MAPSIZE + FRAMEOFFSET)),
                               data.numMaps % DEV_FRAMES);
            DEBUG(offset << "/" << data.numMaps << " frames uploaded");
        }
        // force wait for one device to finish since there's no new data
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
        "time:" + std::to_string(
            (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":running_pedestal0:dev" + std::to_string(nextFree.back()) +
            ".bmp",
        iped,
        0);
    save_image<Photon>(
        "time:" + std::to_string(
            (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":running_pedestal1:dev" + std::to_string(nextFree.back()) +
            ".bmp",
        iped,
        1);
    save_image<Photon>(
        "time:" + std::to_string(
            (std::chrono::duration_cast<ms>((Clock::now() - t))).count()) +
            ":running_pedestal2:dev" + std::to_string(nextFree.back()) +
            ".bmp",
        iped,
        2);

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
        dev->stream,
        dev->data,
        alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                        Data,
                                        typename TAlpaka::Dim,
                                        typename TAlpaka::Size>(
            data, host, (numMaps * (MAPSIZE + FRAMEOFFSET))),
        numMaps * (MAPSIZE + FRAMEOFFSET));

    // copy offset data from last device uploaded to
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
        static_cast<std::string>(std::to_string(dev->id) + "data" +
                                 std::to_string(std::rand() % 1000) + ".bmp"),
        data,
        DEV_FRAMES - 1);

    // the event is used to wait for pedestal data
    alpaka::stream::enqueue(dev->stream, dev->event);

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
    
    alpaka::wait::wait(dev->stream, dev->event);
    
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
