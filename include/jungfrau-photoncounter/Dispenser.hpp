#pragma once

#include "Alpakaconfig.hpp"
#include "Ringbuffer.hpp"
#include "deviceData.hpp"

#include "kernel/Calibration.hpp"
#include "kernel/ClusterFinder.hpp"
#include "kernel/Conversion.hpp"
#include "kernel/DriftMap.hpp"
#include "kernel/GainStageMasking.hpp"
#include "kernel/GainmapInversion.hpp"
#include "kernel/MaxValueCopy.hpp"
#include "kernel/PhotonFinder.hpp"
#include "kernel/Reduction.hpp"
#include "kernel/Summation.hpp"

#include <boost/optional.hpp>

#include <iostream>
#include <limits>

template <typename TAlpaka, typename TDim, typename TSize> class Dispenser {
public:
    /**
     * Dispenser constructor
     * @param Maps-Struct with initial gain
     */
    Dispenser(
        FramePackage<GainMap, TAlpaka, TDim, TSize> gainMap,
        boost::optional<
            alpaka::mem::buf::
                Buf<typename TAlpaka::DevHost, MaskMap, TDim, TSize>> mask)
        : host(alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u)),
          gain(gainMap),
          mask((
              mask ? *mask
                   : alpaka::mem::buf::alloc<MaskMap, TSize>(host, SINGLEMAP))),
          drift(alpaka::mem::buf::alloc<DriftMap, TSize>(host, SINGLEMAP)),
          gainStage(
              alpaka::mem::buf::alloc<GainStageMap, TSize>(host, SINGLEMAP)),
          maxValueMaps(
              alpaka::mem::buf::alloc<EnergyMap, TSize>(host, DEV_FRAMES)),
          init(false),
          ringbuffer(TAlpaka::STREAMS_PER_DEV *
                     alpaka::pltf::getDevCount<typename TAlpaka::PltfAcc>()),
          pedestal(PEDEMAPS, host)
    {
        std::vector<typename TAlpaka::DevAcc> allDevices(
            alpaka::pltf::getDevs<typename TAlpaka::PltfAcc>());

        initDevices(allDevices);

        // make room for live mask information
        if (!mask) {
            alpaka::mem::view::set(devices[devices.size() - 1].queue,
                                   devices[devices.size() - 1].mask,
                                   1,
                                   SINGLEMAP);
        }

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
                         << " pedestalframes uploaded (1)");
        }

        // upload remaining frames
        if (offset != data.numFrames) {
            offset += calcPedestaldata(
                alpaka::mem::view::getPtrNative(data.data) + offset,
                data.numFrames % DEV_FRAMES);
            DEBUG(offset << "/" << data.numFrames
                         << " pedestalframes uploaded (2)");
        }

        distributeMaskMaps();
        distributeInitialPedestalMaps();
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
        auto current_device = devices[nextFree];

        DEBUG("downloading pedestaldata from device " << nextFree);

        // get the pedestal data from the device
        alpaka::mem::view::copy(current_device.queue,
                                pedestal.data,
                                current_device.pedestal,
                                PEDEMAPS);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue);

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
        auto current_device = devices[nextFree];

        // get the pedestal data from the device
        alpaka::mem::view::copy(
            current_device.queue, &mask, current_device.mask, SINGLEMAP);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue);

        return mask;
    }

    /**
     * Flags alldevices as ready for download.
     */
    auto flush() -> void
    {
        DEBUG("flushing...");
        synchronize();
        for (auto& device : devices)
            if (device.state != FREE)
                device.state = READY;
    }
    /**
     * Downloads the current gain stage map.
     * @return gain stage map
     */
    auto downloadGainStages(std::size_t frame = 0) -> GainStageMap*
    {
        DEBUG("downloading gain stage map...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree];

        // mask gain stage maps
        GainStageMaskingKernel gainStageMasking;
        auto const gainStageMasker(
            alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                getWorkDiv<TAlpaka>(),
                gainStageMasking,
                alpaka::mem::view::getPtrNative(current_device.gainStage),
                alpaka::mem::view::getPtrNative(current_device.gainStageOutput),
                frame,
                alpaka::mem::view::getPtrNative(current_device.mask)));

        alpaka::queue::enqueue(current_device.queue, gainStageMasker);

        // get the pedestal data from the device
        alpaka::mem::view::copy(current_device.queue,
                                gainStage,
                                current_device.gainStageOutput,
                                SINGLEMAP);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue);

        return alpaka::mem::view::getPtrNative(gainStage);
    }

    /**
     * Downloads the current drift map.
     * @return drift map
     */
    auto downloadDriftMaps() -> DriftMap*
    {
        DEBUG("downloading drift map...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree];

        // mask gain stage maps
        DriftMapKernel driftMapKernel;
        auto const driftMap(
            alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                getWorkDiv<TAlpaka>(),
                driftMapKernel,
                alpaka::mem::view::getPtrNative(current_device.initialPedestal),
                alpaka::mem::view::getPtrNative(current_device.pedestal),
                alpaka::mem::view::getPtrNative(current_device.drift)));

        alpaka::queue::enqueue(current_device.queue, driftMap);

        // get the pedestal data from the device
        alpaka::mem::view::copy(
            current_device.queue, drift, current_device.drift, SINGLEMAP);

        // wait for copy to finish
        alpaka::wait::wait(current_device.queue);

        return alpaka::mem::view::getPtrNative(drift);
    }

    /**
     * Tries to upload one data package.
     * @param Maps-struct with raw data, offset within the package
     * @return number of frames uploaded from the package
     */
    auto uploadData(FramePackage<DetectorData, TAlpaka, TDim, TSize> data,
                    std::size_t offset,
                    ExecutionFlags flags) -> std::size_t
    {
        if (!ringbuffer.isEmpty()) {
            // try uploading one data package
            if (offset <= data.numFrames - DEV_FRAMES) {
                offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                       offset,
                                   DEV_FRAMES,
                                   flags);
                DEBUG(offset << "/" << data.numFrames << " frames uploaded");
            }
            // upload remaining frames
            else if (offset != data.numFrames) {
                offset += calcData(alpaka::mem::view::getPtrNative(data.data) +
                                       offset,
                                   data.numFrames % DEV_FRAMES,
                                   flags);
                DEBUG(offset << "/" << data.numFrames << " frames uploaded");
            }
            // force wait for one device to finish since there's no new data
            else {
                //! @todo: CHECK THIS!!!
                //! @todo: does this conflict with the other flush????
                //! @todo: does this concept even make sense???
                uint64_t currentFull =
                    (nextFull + devices.size() - 1) % devices.size();
                DEBUG("flushing stuff ... (" << currentFull << ")");
                alpaka::wait::wait(devices[currentFull].queue,
                                   devices[currentFull].event);
                devices[currentFull].state = READY;
            }
        }

        return offset;
    }

    /**
     * Tries to download one data package.
     * @param pointer to empty struct for photon and sum maps and cluster data
     * @return boolean indicating whether maps were downloaded or not
     */
    auto downloadData(
        boost::optional<FramePackage<EnergyMap, TAlpaka, TDim, TSize>&> energy,
        boost::optional<FramePackage<PhotonMap, TAlpaka, TDim, TSize>&> photon,
        boost::optional<FramePackage<EnergySumMap, TAlpaka, TDim, TSize>&> sum,
        boost::optional<FramePackage<EnergyValue, TAlpaka, TDim, TSize>&>
            maxValues,
        boost::optional<ClusterArray<TAlpaka, TDim, TSize>&> clusters) -> size_t
    {
        struct DeviceData<TAlpaka, TDim, TSize>* dev =
            &Dispenser::devices[nextFree];


        DEBUG("downloading from device " << nextFree << " (nextFull is "
                                         << nextFull << ")");
        std::string s = "";
        for (const auto& d : devices) {
            s += (d.state == READY ? "READY" : "");
            s += (d.state == FREE ? "FREE" : "");
            s += (d.state == PROCESSING ? "PROCESSING" : "");
            s += " ";
        }
        DEBUG("Current state: " << s);


        // to keep frames in order only download if the longest running device
        // has finished
        if (dev->state != READY)
            return 0;

        // download energy if needed
        if (energy) {
            DEBUG("downloading energy");
            (*energy).numFrames = dev->numMaps;
            alpaka::mem::view::copy(
                dev->queue, (*energy).data, dev->energy, dev->numMaps);
        }

        // download photon data if needed
        if (photon) {
            DEBUG("downloading photons");
            (*photon).numFrames = dev->numMaps;
            alpaka::mem::view::copy(
                dev->queue, (*photon).data, dev->photon, dev->numMaps);
        }

        // download summation frames if needed
        if (sum) {
            DEBUG("downloading sum");
            (*sum).numFrames = dev->numMaps / SUM_FRAMES;
            alpaka::mem::view::copy(
                dev->queue, (*sum).data, dev->sum, (dev->numMaps / SUM_FRAMES));
        }

        // download maximum values if needed
        if (maxValues) {
            DEBUG("downloading max values");
            (*maxValues).numFrames = dev->numMaps;
            alpaka::mem::view::copy(
                dev->queue, (*maxValues).data, dev->maxValues, DEV_FRAMES);
        }

        // download number of clusters if needed
        if (clusters) {
            DEBUG("downloading clusters");
            alpaka::mem::view::copy(dev->queue,
                                    (*clusters).usedPinned,
                                    dev->numClusters,
                                    SINGLEMAP);

            // wait for completion of copy operations
            alpaka::wait::wait(dev->queue);

            // reserve space in the cluster array for the new data and save the
            // number of clusters to download temporarily
            auto oldNumClusters = (*clusters).used;
            auto clustersToDownload =
                alpaka::mem::view::getPtrNative((*clusters).usedPinned)[0];
            alpaka::mem::view::getPtrNative((*clusters).usedPinned)[0] +=
                oldNumClusters;
            (*clusters).used =
                alpaka::mem::view::getPtrNative((*clusters).usedPinned)[0];


            //! @todo: remove this line later
            DEBUG("clustersToDownload: " << clustersToDownload);


            // create a subview in the cluster buffer where the new data shuld
            // be downloaded to
            auto const extentView(alpaka::vec::Vec<TDim, TSize>(
                static_cast<TSize>(clustersToDownload)));
            auto const offsetView(alpaka::vec::Vec<TDim, TSize>(
                static_cast<TSize>(oldNumClusters)));
            alpaka::mem::view::
                ViewSubView<typename TAlpaka::DevHost, Cluster, TDim, TSize>
                    clusterView((*clusters).clusters, extentView, offsetView);

            // download actual clusters
            alpaka::mem::view::copy(
                dev->queue, clusterView, dev->cluster, clustersToDownload);
        }

        alpaka::wait::wait(dev->queue);

        dev->state = FREE;
        nextFree = (nextFree + 1) % devices.size();
        ringbuffer.push(dev);

        return DEV_FRAMES; //(*photon).numFrames;
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
    typename TAlpaka::DevHost host;
    FramePackage<GainMap, TAlpaka, TDim, TSize> gain;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, MaskMap, TDim, TSize> mask;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, DriftMap, TDim, TSize>
        drift;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, GainStageMap, TDim, TSize>
        gainStage;
    alpaka::mem::buf::Buf<typename TAlpaka::DevHost, EnergyMap, TDim, TSize>
        maxValueMaps;

    FramePackage<PedestalMap, TAlpaka, TDim, TSize> pedestal;

    bool init;
    Ringbuffer<DeviceData<TAlpaka, TDim, TSize>*> ringbuffer;
    std::vector<DeviceData<TAlpaka, TDim, TSize>> devices;

    std::size_t nextFree, nextFull;

    /**
     * Initializes all devices. Uploads gain data and creates buffer.
     * @param vector with devices to be initialized
     */
    auto initDevices(std::vector<typename TAlpaka::DevAcc> allDevices) -> void
    {
        const GainmapInversionKernel gainmapInversionKernel;
        devices.reserve(allDevices.size() * TAlpaka::STREAMS_PER_DEV);
        for (std::size_t num = 0; num < allDevices.size() * TAlpaka::STREAMS_PER_DEV;
             ++num) {
            // initialize variables
            devices.emplace_back(num, allDevices[num / TAlpaka::STREAMS_PER_DEV]);
            alpaka::mem::view::copy(
                devices[num].queue, devices[num].gain, gain.data, GAINMAPS);
            // compute reciprocals of gain maps
            auto const gainmapInversion(
                alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                    getWorkDiv<TAlpaka>(),
                    gainmapInversionKernel,
                    alpaka::mem::view::getPtrNative(devices[num].gain)));
            alpaka::queue::enqueue(devices[num].queue, gainmapInversion);

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

        DEBUG("calcPedestaldata on dev " << dev->id);

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
        auto prevDevice = (nextFull + devices.size() - 1) % devices.size();
        alpaka::wait::wait(devices[prevDevice].queue);
        alpaka::mem::view::copy(
            dev->queue, dev->pedestal, devices[prevDevice].pedestal, PEDEMAPS);

        alpaka::mem::view::copy(
            dev->queue, dev->mask, devices[prevDevice].mask, SINGLEMAP);

        // increase nextFull and nextFree (because pedestal data isn't
        // downloaded like normal data)
        nextFull = (nextFull + 1) % devices.size();
        nextFree = (nextFree + 1) % devices.size();

        if (init == false) {
            alpaka::mem::view::set(dev->queue, dev->pedestal, 0, SINGLEMAP);
            alpaka::wait::wait(dev->queue);
            init = true;
        }

        CalibrationKernel calibrationKernel;
        auto const calibration(
            alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                getWorkDiv<TAlpaka>(),
                calibrationKernel,
                alpaka::mem::view::getPtrNative(dev->data),
                alpaka::mem::view::getPtrNative(dev->pedestal),
                alpaka::mem::view::getPtrNative(dev->mask),
                dev->numMaps));

        alpaka::queue::enqueue(dev->queue, calibration);

        alpaka::wait::wait(dev->queue);

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
        uint64_t source = (nextFull + devices.size() - 1) % devices.size();
        DEBUG("distributeMaskMaps (from " << source << ")");
        for (uint64_t i = 0; i < devices.size() - 1; ++i) {
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
     * Distributes copies the initial pedestal map of the current accelerator to
     * all others.
     */
    auto distributeInitialPedestalMaps() -> void
    {
        uint64_t source = (nextFull + devices.size() - 1) % devices.size();
        DEBUG("distributeInitialPedestalMaps (from " << source << ")");
        for (uint64_t i = 0; i < devices.size(); ++i) {
            alpaka::mem::view::copy(devices[source].queue,
                                    devices[i].initialPedestal,
                                    devices[source].pedestal,
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
    auto calcData(TDetectorData* data,
                  std::size_t numMaps,
                  ExecutionFlags flags) -> std::size_t
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
        auto prevDevice = (nextFull + devices.size() - 1) % devices.size();
        alpaka::wait::wait(dev->queue, devices[prevDevice].event);
        DEBUG("device " << devices[prevDevice].id << " finished");

        devices[prevDevice].state = READY;
        alpaka::mem::view::copy(
            dev->queue, dev->pedestal, devices[prevDevice].pedestal, PEDEMAPS);
        nextFull = (nextFull + 1) % devices.size();

        // reset the number of clusters
        alpaka::mem::view::set(dev->queue, dev->numClusters, 0, SINGLEMAP);

        MaskMap* local_mask = flags.masking
                                  ? alpaka::mem::view::getPtrNative(dev->mask)
                                  : nullptr;


        // converting to energy
        // the photon and cluster extraction kernels already include energy
        // conversion
        if (flags.mode == 0) {
            ConversionKernel conversionKernel;
            auto const conversion(
                alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                    getWorkDiv<TAlpaka>(),
                    conversionKernel,
                    alpaka::mem::view::getPtrNative(dev->data),
                    alpaka::mem::view::getPtrNative(dev->gain),
                    alpaka::mem::view::getPtrNative(dev->pedestal),
                    alpaka::mem::view::getPtrNative(dev->gainStage),
                    alpaka::mem::view::getPtrNative(dev->energy),
                    dev->numMaps,
                    local_mask));

            DEBUG("enqueueing conversion kernel");
            alpaka::queue::enqueue(dev->queue, conversion);
        }

        // converting to photons (and energy)
        if (flags.mode == 1) {
            PhotonFinderKernel photonFinderKernel;
            auto const photonFinder(
                alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                    getWorkDiv<TAlpaka>(),
                    photonFinderKernel,
                    alpaka::mem::view::getPtrNative(dev->data),
                    alpaka::mem::view::getPtrNative(dev->gain),
                    alpaka::mem::view::getPtrNative(dev->pedestal),
                    alpaka::mem::view::getPtrNative(dev->gainStage),
                    alpaka::mem::view::getPtrNative(dev->energy),
                    alpaka::mem::view::getPtrNative(dev->photon),
                    dev->numMaps,
                    local_mask));

            DEBUG("enqueueing photon kernel");
            alpaka::queue::enqueue(dev->queue, photonFinder);
        };

        // clustering (and conversion to energy)
        if (flags.mode == 2) {
            DEBUG("enqueueing clustering kernel");

            for (uint32_t i = 0; i < numMaps + 1; ++i) {
                // execute the clusterfinder with the pedestalupdate on every
                // frame
                ClusterFinderKernel clusterFinderKernel;
                auto const clusterFinder(
                    alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                        getWorkDiv<TAlpaka>(),
                        clusterFinderKernel,
                        alpaka::mem::view::getPtrNative(dev->data),
                        alpaka::mem::view::getPtrNative(dev->gain),
                        alpaka::mem::view::getPtrNative(dev->pedestal),
                        alpaka::mem::view::getPtrNative(dev->gainStage),
                        alpaka::mem::view::getPtrNative(dev->energy),
                        alpaka::mem::view::getPtrNative(dev->cluster),
                        alpaka::mem::view::getPtrNative(dev->numClusters),
                        local_mask,
                        dev->numMaps,
                        i));

                alpaka::queue::enqueue(dev->queue, clusterFinder);
            }
        }

        // find max value
        if (flags.maxValue) {
            // get the max value
            for (uint32_t i = 0; i < numMaps; ++i) {
                // reduce all images
                WorkDiv workdivRun1{TAlpaka::blocksPerGrid,
                                    TAlpaka::threadsPerBlock,
                                    static_cast<Size>(1)};
                ReduceKernel<TAlpaka::threadsPerBlock, double> reduceKernelRun1;
                auto const reduceRun1(
                    alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                        workdivRun1,
                        reduceKernelRun1,
                        &alpaka::mem::view::getPtrNative(dev->energy)[i],
                        &alpaka::mem::view::getPtrNative(dev->maxValueMaps)[i],
                        DIMX * DIMY));

                WorkDiv workdivRun2{static_cast<Size>(1),
                                    TAlpaka::threadsPerBlock,
                                    static_cast<Size>(1)};
                ReduceKernel<TAlpaka::threadsPerBlock, double> reduceKernelRun2;
                auto const reduceRun2(
                    alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                        workdivRun2,
                        reduceKernelRun2,
                        &alpaka::mem::view::getPtrNative(dev->maxValueMaps)[i],
                        &alpaka::mem::view::getPtrNative(dev->maxValueMaps)[i],
                        TAlpaka::blocksPerGrid));
                alpaka::queue::enqueue(dev->queue, reduceRun1);
                alpaka::queue::enqueue(dev->queue, reduceRun2);
            }

            WorkDiv workdivMaxValueCopy{
                static_cast<Size>(
                    std::ceil((double)numMaps / TAlpaka::threadsPerBlock)),
                TAlpaka::threadsPerBlock,
                static_cast<Size>(1)};
            MaxValueCopyKernel maxValueCopyKernel;
            auto const maxValueCopy(
                alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                    workdivMaxValueCopy,
                    maxValueCopyKernel,
                    alpaka::mem::view::getPtrNative(dev->maxValueMaps),
                    alpaka::mem::view::getPtrNative(dev->maxValues),
                    numMaps));

            DEBUG("enqueueing max value extraction kernel");
            alpaka::queue::enqueue(dev->queue, maxValueCopy);
        }

        // summation
        if (flags.summation) {
            DEBUG("enqueueing summation kernel");

            SummationKernel summationKernel;
            auto const summation(
                alpaka::kernel::createTaskKernel<typename TAlpaka::Acc>(
                    getWorkDiv<TAlpaka>(),
                    summationKernel,
                    alpaka::mem::view::getPtrNative(dev->energy),
                    SUM_FRAMES,
                    dev->numMaps,
                    alpaka::mem::view::getPtrNative(dev->sum)));

            alpaka::queue::enqueue(dev->queue, summation);
        };

        // the event is used to wait for pedestal data
        alpaka::queue::enqueue(dev->queue, dev->event);

        return numMaps;
    }
};
