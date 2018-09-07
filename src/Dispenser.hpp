#pragma once

#include "Config.hpp"
#include "Ringbuffer.hpp"
#include "deviceData.hpp"

#include "kernel/Correction.hpp"
#include "kernel/Statistics.hpp"
#include "kernel/Summation.hpp"
#include "kernel/Zero.hpp"

#include <iostream>
#include <limits>
#include <mutex>

template <typename TAlpaka> class Dispenser {
public:
    /**
     * Dispenser constructor
     * @param Maps-Struct with initial gain
     */
    Dispenser(Maps<GainMap, TAlpaka> gainMap, Maps<Mask, TAlpaka> mask)
        : gain(gainMap),
          mask(mask),
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
        pedestal.data =
            alpaka::mem::buf::alloc<Pedestal, typename TAlpaka::Size>(host,
                                                                      PEDEMAPS);
        
        // make room for live mask information
        mask.numMaps = SINGLEMAP;
        mask.data =
            alpaka::mem::buf::alloc<Mask, typename TAlpaka::Size>(host,
                                                                      SINGLEMAP);
        
        // make room for live drift information
        drift.numMaps = SINGLEMAP;
        drift.data =
            alpaka::mem::buf::alloc<Drift, typename TAlpaka::Size>(host,
                                                                      SINGLEMAP);
        
        // make room for live gain stage information
        gainStage.numMaps = SINGLEMAP;
        gainStage.data =
            alpaka::mem::buf::alloc<GainStage, typename TAlpaka::Size>(host,
                                                                      SINGLEMAP);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#if (SHOW_DEBUG == false)
        alpaka::mem::buf::pin(pedestal.data);
        alpaka::mem::buf::pin(mask.data);
        alpaka::mem::buf::pin(drift.data);
        alpaka::mem::buf::pin(gainStage.data);
#endif
#endif
    }

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
    auto synchronize() -> void
    {
        for (struct DeviceData<TAlpaka> dev : devices)
            alpaka::wait::wait(dev.queue);
    }

    /**
     * Tries to upload all data packages requiered for the inital offset.
     * Only stops after all data packages are uploaded.
     * @param Maps-Struct with datamaps
     */
    auto uploadPedestaldata(Maps<DetectorData, TAlpaka> data) -> void
    {
        std::size_t offset = 0;
        DEBUG("uploading pedestaldata...");

        // upload all frames cut into smaller packages
        while (offset <= data.numMaps - DEV_FRAMES) {
            offset += calcPedestaldata(
                alpaka::mem::view::getPtrNative(data.data) + offset,
                DEV_FRAMES);
            DEBUG(offset << "/" << data.numMaps << " pedestalframes uploaded");
        }

        // upload remaining frames
        if (offset != data.numMaps) {
            offset += calcPedestaldata(
                alpaka::mem::view::getPtrNative(data.data) + offset,
                data.numMaps % DEV_FRAMES);
            DEBUG(offset << "/" << data.numMaps << " pedestalframes uploaded");
        }
    }

    /**
     * Downloads the pedestal data.
     * @return pedestal pedestal data
     */
    auto downloadPedestaldata() -> Maps<PedestalMap, TAlpaka>
    {
        DEBUG("downloading pedestaldata...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree.back()];

        // get the pedestal data from the device
        alpaka::mem::view::copy(current_device.queue,
                                pedestal.data,
                                current_device.pedestal,
                                PEDEMAPS);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue, current_device.event);

        return pedestal;
    }

    /**
     * Downloads the current mask map.
     * @return mask map
     */
    auto downloadMask() -> Maps<Mask, TAlpaka>;
    {
        DEBUG("downloading mask...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree.back()];

        // get the pedestal data from the device
        alpaka::mem::view::copy(
            current_device.queue, mask.data, current_device.mask, 1);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue, current_device.event);

        return mask;
    }

    /**
     * Downloads the current gain stage map.
     * @return gain stage map
     */
    auto downloadGainStages() -> Maps<GainStageMap, TAlpaka>;
    {
        DEBUG("downloading gain stage map...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree.back()];

        // get the pedestal data from the device
        alpaka::mem::view::copy(
            current_device.queue, gainStage.data, current_device.gainStage, 1);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue, current_device.event);

        return gainStage;
    }

    /**
     * Downloads the current drift map.
     * @return drift map
     */
    auto downloadDriftMaps() -> Maps<Drift, TAlpaka>
    {
        DEBUG("downloading drift map...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree.back()];

        // get the pedestal data from the device
        alpaka::mem::view::copy(
            current_device.queue, drift.data, current_device.drift, 1);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue, current_device.event);

        return drift;
    }

    /**
     * Tries to upload one data package.
     * @param Maps-struct with raw data, offset within the package
     * @return number of frames uploaded from the package
     */
    auto uploadData(Maps<DetectorData, 
                    TAlpaka> data, 
                    std::size_t offset) -> std::size_t
    {
        if (!ringbuffer.isEmpty()) {
            // try uploading one data package
            if (offset <= data.numMaps - DEV_FRAMES) {
                offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                       offset,
                                   DEV_FRAMES);
                DEBUG(offset << "/" << data.numMaps << " frames uploaded");
            }
            // upload remaining frames
            else if (offset != data.numMaps) {
                offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                       offset,
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

    /**
     * Tries to download one data package.
     * @param pointer to empty struct for photon and sum maps
     * @return boolean indicating whether maps were downloaded or not
     */
    auto downloadData(Maps<PhotonMaps, TAlpaka>* photon,
                      Maps<PhotonSum, TAlpaka>* sum) -> bool
    {
        std::lock_guard<std::mutex> lock(mutex);
        struct DeviceData<TAlpaka>* dev = &Dispenser::devices[nextFree.front()];

        // to keep frames in order only download if the longest running device
        // has finished
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


        //
        //
        //! @TODO: debugging
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
        //
        //
        //

        return true;
    }

    /**
     * Returns the a vector with the amount of memory of each device.
     * @return size_array
     */
    auto getMemSize() -> std::vector<std::size_t>
    {
        std::vector<std::size_t> sizes(devices.size());
        for (std::size_t i = 0; i < devices.size(); ++i) {
            sizes[i] = alpaka::dev::getMemBytes(devices[i].device);
        }

        return sizes;
    }

    /**
     * Returns the a vector with the amount of free memory of each device.
     * @return size_array
     */
    auto getFreeMem() -> std::vector<std::size_t>
    {
        std::vector<std::size_t> sizes(devices.size());
        for (std::size_t i = 0; i < devices.size(); ++i) {
            sizes[i] = alpaka::dev::getFreeMemBytes(devices[i].device);
        }

        return sizes;
    }

private:
    Maps<Gain, TAlpaka> gain;
    Maps<Mask, TAlpaka> mask;
    Maps<Drift, TAlpaka> drift;
    Maps<GainStage, TAlpaka> gainStage;
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
    auto initDevices(std::vector<typename TAlpaka::DevAcc> devs) -> void
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
            devices[num].drift =
                alpaka::mem::buf::alloc<Drift, typename TAlpaka::Size>(
                    devs[num / workdiv.STREAMS_PER_DEV], SINGLEMAP);
            devices[num].gainStage =
                alpaka::mem::buf::alloc<GainStage, typename TAlpaka::Size>(
                    devs[num / workdiv.STREAMS_PER_DEV], SINGLEMAP);
            devices[num].maxValue =
                alpaka::mem::buf::alloc<Value, typename TAlpaka::Size>(
                    devs[num / workdiv.STREAMS_PER_DEV], SINGLEMAP);
            devices[num].photon =
                alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
                    devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
            devices[num].sum =
                alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(
                    devs[num / workdiv.STREAMS_PER_DEV],
                    DEV_FRAMES / SUM_FRAMES);
            devices[num].photonHost =
                alpaka::mem::buf::alloc<Photon, typename TAlpaka::Size>(
                    host, DEV_FRAMES);
            devices[num].sumHost =
                alpaka::mem::buf::alloc<PhotonSum, typename TAlpaka::Size>(
                    host, (DEV_FRAMES / SUM_FRAMES));
            devices[num].maxValueHost =
                alpaka::mem::buf::alloc<Value, typename TAlpaka::Size>(
                    host, SINGLEMAP);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#if (SHOW_DEBUG == false)
            // pin all buffer
            alpaka::mem::buf::pin(devices[num].data);
            alpaka::mem::buf::pin(devices[num].gain);
            alpaka::mem::buf::pin(devices[num].pedestal);
            alpaka::mem::buf::pin(devices[num].mask);
            alpaka::mem::buf::pin(devices[num].drift);
            alpaka::mem::buf::pin(devices[num].gainStage);
            alpaka::mem::buf::pin(devices[num].maxValue);
            alpaka::mem::buf::pin(devices[num].photon);
            alpaka::mem::buf::pin(devices[num].sum);
            alpaka::mem::buf::pin(devices[num].photonHost);
            alpaka::mem::buf::pin(devices[num].sumHost);
            alpaka::mem::buf::pin(devices[num].maxValueHost);
#endif
#endif

            if (!ringbuffer.push(&devices[num])) {
                fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
                exit(EXIT_FAILURE);
            }
            DEBUG("Device # " << (num + 1) << " init");
        }
    }

    /**
     * Executes the calibration kernel.
     * @param pointer to raw data and number of frames
     * @return number of frames calculated
     */
    auto calcPedestaldata(Data* data, std::size_t numMaps) -> std::size_t
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
            //! @todo: test this memset
            alpaka::mem::view::set(
                alpaka::mem::view::getPtrNative(dev->pedestal),
                0,
                1,
                dev->queue);

            alpaka::wait::wait(dev->queue);

            init = true;
        }


        //! @todo: use new kernels
        StatisticsKernel StatisticsKernel;
        auto const statistics(
            alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
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

        return numMaps;
    }

    /**
     * Executes summation and correction kernel.
     * @param pointer to raw data and number of frames
     * @return number of frames calculated
     */
    auto calcData(Data* data, std::size_t numMaps) -> std::size_t
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

        // copy offset data from last device uploaded to
        std::lock_guard<std::mutex> lock(mutex);
        alpaka::wait::wait(dev->queue, devices[nextFree.back()].event);
        DEBUG("device " << devices[nextFree.back()].id << " finished");

        devices[nextFree.back()].state = READY;
        alpaka::mem::view::copy(dev->queue,
                                dev->pedestal,
                                devices[nextFree.back()].pedestal,
                                PEDEMAPS);
        nextFree.push_back(dev->id);


        //! @todo: use new kernels
        StatisticsKernel statisticsKernel;
        auto const statistics(
            alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
                workdiv.workdiv,
                statisticsKernel,
                alpaka::mem::view::getPtrNative(dev->data),
                dev->numMaps,
                alpaka::mem::view::getPtrNative(dev->pedestal),
                alpaka::mem::view::getPtrNative(dev->mask)));

        alpaka::queue::enqueue(dev->queue, statistics);
        alpaka::wait::wait(dev->queue);

        CorrectionKernel correctionKernel;
        auto const correction(
            alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
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
        auto const summation(
            alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
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
};
