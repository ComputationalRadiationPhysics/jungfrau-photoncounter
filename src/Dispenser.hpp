#pragma once

#include "Config.hpp"
#include "Ringbuffer.hpp"
#include "deviceData.hpp"

#include "kernel/Calibration.hpp"
#include "kernel/ClusterFinder.hpp"
#include "kernel/Conversion.hpp"
#include "kernel/PhotonFinder.hpp"
#include "kernel/Summation.hpp"

#include <iostream>
#include <limits>
#include <mutex>

template <typename TAlpaka, typename TDim, typename TSize> class Dispenser {
public:
    /**
     * Dispenser constructor
     * @param Maps-Struct with initial gain
     */
    Dispenser(
        FramePackage<GainMap, TAlpaka, TDim, TSize> gainMap,
        alpaka::mem::buf::Buf<typename TAlpaka::DevHost, MaskMap, TDim, TSize>
            mask)
        : gain(gainMap),
          mask(mask),
          drift(alpaka::mem::buf::alloc<DriftMap, TSize>(
              alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
              static_cast<TSize>(0u))),
          gainStage(alpaka::mem::buf::alloc<GainStageMap, TSize>(
              alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
              static_cast<TSize>(0u))),
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

        auto host = alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u);

        // make room for live pedestal information
        pedestal.numFrames = PEDEMAPS;
        pedestal.data =
            alpaka::mem::buf::alloc<PedestalMap, TSize>(host, PEDEMAPS);

        // make room for live mask information
        if (!alpaka::mem::view::getPtrNative(this->mask)) {
            this->mask =
                alpaka::mem::buf::alloc<MaskMap, TSize>(host, SINGLEMAP);
            alpaka::mem::view::set(devices[0].queue, devices[0].mask, 1, SINGLEMAP);
        }

        // make room for live drift information
        drift = alpaka::mem::buf::alloc<DriftMap, TSize>(host, SINGLEMAP);

        // make room for live gain stage information
        gainStage =
            alpaka::mem::buf::alloc<GainStageMap, TSize>(host, SINGLEMAP);

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#if (SHOW_DEBUG == false)
        alpaka::mem::buf::pin(pedestal);
        alpaka::mem::buf::pin(mask);
        alpaka::mem::buf::pin(drift);
        alpaka::mem::buf::pin(gainStage);
#endif
#endif
        synchronize();
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
        for (struct DeviceData<TAlpaka, TDim, TSize> dev : devices)
            alpaka::wait::wait(dev.queue);
    }

    /**
     * Tries to upload all data packages requiered for the inital offset.
     * Only stops after all data packages are uploaded.
     * @param Maps-Struct with datamaps
     */
    auto
    uploadPedestaldata(FramePackage<DetectorData, TAlpaka, TDim, TSize> data)
        -> void
    {
        std::size_t offset = 0;
        DEBUG("uploading pedestaldata...");

        // upload all frames cut into smaller packages
        while (offset <= data.numFrames - DEV_FRAMES) {
            offset += calcPedestaldata(
                alpaka::mem::view::getPtrNative(data.data) + offset,
                DEV_FRAMES);
            DEBUG(offset << "/" << data.numFrames
                         << " pedestalframes uploaded");
        }

        // upload remaining frames
        if (offset != data.numFrames) {
            offset += calcPedestaldata(
                alpaka::mem::view::getPtrNative(data.data) + offset,
                data.numFrames % DEV_FRAMES);
            DEBUG(offset << "/" << data.numFrames
                         << " pedestalframes uploaded");
        }

        distributeMaskMaps();
    }

    /**
     * Downloads the pedestal data.
     * @return pedestal pedestal data
     */
    auto downloadPedestaldata()
        -> FramePackage<PedestalMap, TAlpaka, TDim, TSize>
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

        pedestal.numFrames = PEDEMAPS;

        return pedestal;
    }

    /**
     * Downloads the current mask map.
     * @return mask map
     */
    auto downloadMask() -> MaskMap
    {
        DEBUG("downloading mask...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree.back()];

        // get the pedestal data from the device
        alpaka::mem::view::copy(
            current_device.queue, &mask, current_device.mask, SINGLEMAP);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue, current_device.event);

        return mask;
    }

    /**
     * Downloads the current gain stage map.
     * @return gain stage map
     */
    auto downloadGainStages() -> GainStageMap
    {
        DEBUG("downloading gain stage map...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree.back()];

        // get the pedestal data from the device
        alpaka::mem::view::copy(
            current_device.queue, gainStage, current_device.gainStage, 1);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue, current_device.event);

        return *gainStage;
    }

    /**
     * Downloads the current drift map.
     * @return drift map
     */
    auto downloadDriftMaps() -> DriftMap
    {
        DEBUG("downloading drift map...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree.back()];

        // get the pedestal data from the device
        alpaka::mem::view::copy(
            current_device.queue, drift, current_device.drift, 1);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue, current_device.event);

        return *drift;
    }

    /**
     * Tries to upload one data package.
     * @param Maps-struct with raw data, offset within the package
     * @return number of frames uploaded from the package
     */
    auto uploadData(FramePackage<DetectorData, TAlpaka, TDim, TSize> data,
                    std::size_t offset) -> std::size_t
    {
        if (!ringbuffer.isEmpty()) {
            // try uploading one data package
            if (offset <= data.numFrames - DEV_FRAMES) {
                offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                       offset,
                                   DEV_FRAMES);
                DEBUG(offset << "/" << data.numFrames << " frames uploaded");
            }
            // upload remaining frames
            else if (offset != data.numFrames) {
                offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                       offset,
                                   data.numFrames % DEV_FRAMES);
                DEBUG(offset << "/" << data.numFrames << " frames uploaded");
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
    auto downloadData(FramePackage<PhotonMap, TAlpaka, TDim, TSize>* photon,
                      FramePackage<PhotonSumMap, TAlpaka, TDim, TSize>* sum)
        -> bool
    {
        std::lock_guard<std::mutex> lock(mutex);
        struct DeviceData<TAlpaka, TDim, TSize>* dev =
            &Dispenser::devices[nextFree.front()];

        // to keep frames in order only download if the longest running device
        // has finished
        if (dev->state != READY)
            return false;

        photon->numFrames = dev->numMaps;
        alpaka::mem::view::copy(
            dev->queue, dev->photonHost, dev->photon, dev->numMaps);
        photon->data = dev->photonHost;

        sum->numFrames = dev->numMaps / SUM_FRAMES;
        alpaka::mem::view::copy(
            dev->queue, dev->sumHost, dev->sum, (dev->numMaps / SUM_FRAMES));
        sum->data = dev->sumHost;

        alpaka::wait::wait(dev->queue, dev->event);

        dev->state = FREE;
        nextFree.pop_front();
        ringbuffer.push(dev);
        DEBUG("device " << dev->id << " freed");

        /*
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
        */

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
    FramePackage<GainMap, TAlpaka, TDim, TSize> gain;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, MaskMap, TDim, TSize> mask;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, DriftMap, TDim, TSize>
        drift;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, GainStageMap, TDim, TSize>
        gainStage;
    FramePackage<PedestalMap, TAlpaka, TDim, TSize> pedestal;

    TAlpaka workdiv;
    bool init;
    Ringbuffer<DeviceData<TAlpaka, TDim, TSize>*> ringbuffer;
    std::vector<DeviceData<TAlpaka, TDim, TSize>> devices;
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
            devices[num].data = alpaka::mem::buf::alloc<DetectorData, TSize>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
            devices[num].gain = alpaka::mem::buf::alloc<GainMap, TSize>(
                devs[num / workdiv.STREAMS_PER_DEV], GAINMAPS);
            alpaka::mem::view::copy(
                devices[num].queue, devices[num].gain, gain.data, GAINMAPS);
            devices[num].pedestal = alpaka::mem::buf::alloc<PedestalMap, TSize>(
                devs[num / workdiv.STREAMS_PER_DEV], PEDEMAPS);
            devices[num].mask = alpaka::mem::buf::alloc<MaskMap, TSize>(
                devs[num / workdiv.STREAMS_PER_DEV], SINGLEMAP);
            devices[num].drift = alpaka::mem::buf::alloc<DriftMap, TSize>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
            devices[num].energy = alpaka::mem::buf::alloc<EnergyMap, TSize>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
            devices[num].gainStage =
                alpaka::mem::buf::alloc<GainStageMap, TSize>(
                    devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
            devices[num].maxValue = alpaka::mem::buf::alloc<EnergyMap, TSize>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
            devices[num].photon = alpaka::mem::buf::alloc<PhotonMap, TSize>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES);
            devices[num].sum = alpaka::mem::buf::alloc<PhotonSumMap, TSize>(
                devs[num / workdiv.STREAMS_PER_DEV], DEV_FRAMES / SUM_FRAMES);
            devices[num].photonHost =
                alpaka::mem::buf::alloc<PhotonMap, TSize>(host, DEV_FRAMES);
            devices[num].sumHost = alpaka::mem::buf::alloc<PhotonSumMap, TSize>(
                host, (DEV_FRAMES / SUM_FRAMES));
            devices[num].energyHost =
                alpaka::mem::buf::alloc<EnergyMap, TSize>(host, DEV_FRAMES);
            devices[num].maxValueHost =
                alpaka::mem::buf::alloc<EnergyMap, TSize>(host, SINGLEMAP);

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
#endif
#endif
            // copy mask input data on to GPUs
            alpaka::mem::view::copy(
                devices[num].queue, devices[num].mask, mask, PEDEMAPS);


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
    template <typename TDetectorData>
    auto calcPedestaldata(TDetectorData* data, std::size_t numMaps)
        -> std::size_t
    {
        DeviceData<TAlpaka, TDim, TSize>* dev;
        if (!ringbuffer.pop(dev))
            return 0;

        dev->state = PROCESSING;
        dev->numMaps = numMaps;

        alpaka::mem::view::copy(
            dev->queue,
            dev->data,
            alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                            TDetectorData,
                                            TDim,
                                            TSize>(data, host, numMaps),
            numMaps);

        // copy offset data from last initialized device
        std::lock_guard<std::mutex> lock(mutex);
        if (nextFree.size() > 0) {
            alpaka::wait::wait(devices[nextFree.back()].queue);
            alpaka::mem::view::copy(dev->queue,
                                    dev->pedestal,
                                    devices[nextFree.back()].pedestal,
                                    PEDEMAPS);

            alpaka::mem::view::copy(
                dev->queue, dev->mask, devices[nextFree.back()].mask, PEDEMAPS);

            nextFree.pop_front();
        }
        nextFree.push_back(dev->id);

        if (init == false) {
            alpaka::mem::view::set(dev->queue, dev->pedestal, 0, SINGLEMAP);
            alpaka::wait::wait(dev->queue);
            init = true;
        }

        CalibrationKernel calibrationKernel;
        auto const calibration(
            alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
                workdiv.workdiv,
                calibrationKernel,
                alpaka::mem::view::getPtrNative(dev->data),
                alpaka::mem::view::getPtrNative(dev->pedestal),
                alpaka::mem::view::getPtrNative(dev->mask),
                dev->numMaps));

        alpaka::queue::enqueue(dev->queue, calibration);

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
     * Distributes copies the mask map of the current accelerator to all others.
     */
    auto distributeMaskMaps() -> void
    {
      //! @todo: check if this works. 
      uint64_t source = nextFree.front();
        for (uint64_t i = 1; i < devices.size(); ++i) {
            uint64_t destination =
              (i + source + (i >= source ? 1 : 0)) % devices.size();
            alpaka::mem::view::copy(devices[source].queue,
                                    devices[destination].mask,
                                    devices[source].mask,
                                    SINGLEMAP);
        }
        synchronize();
    }

    /**
     * Executes summation and correction kernel.
     * @param pointer to raw data and number of frames
     * @return number of frames calculated
     */
    template <typename TDetectorData>
    auto calcData(TDetectorData* data, std::size_t numMaps) -> std::size_t
    {
        DeviceData<TAlpaka, TDim, TSize>* dev;
        if (!ringbuffer.pop(dev))
            return 0;

        dev->state = PROCESSING;
        dev->numMaps = numMaps;

        alpaka::mem::view::copy(
            dev->queue,
            dev->data,
            alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                            DetectorData,
                                            TDim,
                                            TSize>(data, host, numMaps),
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

        ConversionKernel conversionKernel;
        auto const conversion(
            alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
                workdiv.workdiv,
                conversionKernel,
                alpaka::mem::view::getPtrNative(dev->data),
                alpaka::mem::view::getPtrNative(dev->gain),
                alpaka::mem::view::getPtrNative(dev->pedestal),
                alpaka::mem::view::getPtrNative(dev->gainStage),
                alpaka::mem::view::getPtrNative(dev->energy),
                dev->numMaps,
                alpaka::mem::view::getPtrNative(dev->mask)));

        PhotonFinderKernel photonFinderKernel;
        auto const photonFinder(
            alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
                workdiv.workdiv,
                photonFinderKernel,
                alpaka::mem::view::getPtrNative(dev->data),
                alpaka::mem::view::getPtrNative(dev->gain),
                alpaka::mem::view::getPtrNative(dev->pedestal),
                alpaka::mem::view::getPtrNative(dev->gainStage),
                alpaka::mem::view::getPtrNative(dev->energy),
                alpaka::mem::view::getPtrNative(dev->photon),
                dev->numMaps));
        /*
        // TODO: uncomment summation
        SummationKernel summationKernel;
        auto const summation(
            alpaka::kernel::createTaskExec<typename TAlpaka::Acc>(
                workdiv.workdiv,
                summationKernel,
                alpaka::mem::view::getPtrNative(dev->photon),
                SUM_FRAMES,
                dev->numMaps,
                alpaka::mem::view::getPtrNative(dev->sum)));
        */
        alpaka::queue::enqueue(dev->queue, conversion);
        alpaka::queue::enqueue(dev->queue, photonFinder);
        // alpaka::queue::enqueue(dev->queue, summation);


        alpaka::mem::view::copy(
            dev->queue, dev->energyHost, dev->energy, PEDEMAPS);

        alpaka::wait::wait(dev->queue); //! @todo: do we really have to wait????

        save_image<DetectorData>(
            static_cast<std::string>(std::to_string(dev->id) + "data" +
                                     std::to_string(std::rand() % 1000)),
            data,
            DEV_FRAMES - 1);

        save_image<EnergyMap>(
            static_cast<std::string>(std::to_string(dev->id) + "energy" +
                                     std::to_string(std::rand() % 1000)),
            alpaka::mem::view::getPtrNative(dev->energyHost),
            DEV_FRAMES - 1);

        // the event is used to wait for pedestal data
        alpaka::queue::enqueue(dev->queue, dev->event);

        return numMaps;
    }
};
