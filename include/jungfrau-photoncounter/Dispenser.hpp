#pragma once

#include "Alpakaconfig.hpp"
#include "Ringbuffer.hpp"
#include "deviceData.hpp"

#include "kernel/Calibration.hpp"
#include "kernel/CheckStdDev.hpp"
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

template <typename TAlpaka> class Dispenser {
public:
    /**
     * Dispenser constructor
     * @param Maps-Struct with initial gain
     */
    Dispenser(FramePackage<GainMap, TAlpaka> gainMap,
              boost::optional<typename TAlpaka::HostBuf<MaskMap>> mask)
        : host(alpakaGetHost<TAlpaka>()),
          gain(gainMap),
          mask((mask ? *mask : alpakaAlloc<MaskMap>(host, SINGLEMAP))),
          drift(alpakaAlloc<DriftMap>(host, SINGLEMAP)),
          gainStage(alpakaAlloc<GainStageMap>(host, SINGLEMAP)),
          maxValueMaps(alpakaAlloc<EnergyMap>(host, DEV_FRAMES)),
          pedestalFallback(false),
          init(false),
          ringbuffer(TAlpaka::STREAMS_PER_DEV * alpakaGetDevCount<TAlpaka>()),
          pedestal(PEDEMAPS, host),
          initPedestal(PEDEMAPS, host),
          nextFull(0),
          nextFree(0)
    {
        std::vector<typename TAlpaka::DevAcc> allDevices(
            alpakaGetDevs<TAlpaka>());

        initDevices(allDevices);

        // make room for live mask information
        if (!mask) {
            alpakaMemSet(devices[devices.size() - 1].queue,
                         devices[devices.size() - 1].mask,
                         1,
                         SINGLEMAP);
        }
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
        for (struct DeviceData<TAlpaka> dev : devices)
            alpakaWait(dev.queue);
    }

    /**
     * Tries to upload all data packages requiered for the inital offset.
     * Only stops after all data packages are uploaded.
     * @param Maps-Struct with datamaps
     * @param stdDevThreshold An standard deviation threshold above which pixels
     * should be masked. If this is 0, no pixels will be masked.
     */
    auto uploadPedestaldata(FramePackage<DetectorData, TAlpaka> data,
                            double stdDevThreshold = 0) -> void
    {
        std::size_t offset = 0;
        DEBUG("uploading pedestaldata...");

        // upload all frames cut into smaller packages
        while (offset <= data.numFrames - DEV_FRAMES) {
            offset += calcPedestaldata(alpakaNativePtr(data.data) + offset,
                                       DEV_FRAMES);
        }

        // upload remaining frames
        if (offset != data.numFrames) {
            offset += calcPedestaldata(alpakaNativePtr(data.data) + offset,
                                       data.numFrames % DEV_FRAMES);
        }

        if (stdDevThreshold != 0)
            maskStdDevOver(stdDevThreshold);
        distributeMaskMaps();
        distributeInitialPedestalMaps();
    }

    /**
     * Downloads the pedestal data.
     * @return pedestal pedestal data
     */
    auto downloadPedestaldata() -> FramePackage<PedestalMap, TAlpaka>
    {
        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree];

        DEBUG("downloading pedestaldata from device", nextFree);

        // get the pedestal data from the device
        alpakaCopy(current_device.queue,
                   pedestal.data,
                   current_device.pedestal,
                   PEDEMAPS);

        // wait for copy to finish
        alpakaWait(current_device.queue);

        pedestal.numFrames = PEDEMAPS;

        return pedestal;
    }

    /**
     * Downloads the initial pedestal data.
     * @return pedestal pedestal data
     */
    auto downloadInitialPedestaldata() -> FramePackage<InitPedestalMap, TAlpaka>
    {
        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree];

        DEBUG("downloading pedestaldata from device", nextFree);

        // get the pedestal data from the device
        alpakaCopy(current_device.queue,
                   initPedestal.data,
                   current_device.initialPedestal,
                   PEDEMAPS);

        // wait for copy to finish
        alpakaWait(current_device.queue);

        initPedestal.numFrames = PEDEMAPS;

        return initPedestal;
    }

    /**
     * Downloads the current mask map.
     * @return mask map
     */
    auto downloadMask() -> MaskMap*
    {
        DEBUG("downloading mask...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree];

        // get the pedestal data from the device
        alpakaCopy(current_device.queue, mask, current_device.mask, SINGLEMAP);

        // wait for copy to finish
        alpakaWait(current_device.queue);

        return alpakaNativePtr(mask);
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
        auto const gainStageMasker(alpakaCreateKernel<TAlpaka>(
            getWorkDiv<TAlpaka>(),
            gainStageMasking,
            alpakaNativePtr(current_device.gainStage),
            alpakaNativePtr(current_device.gainStageOutput),
            frame,
            alpakaNativePtr(current_device.mask)));

        alpakaEnqueueKernel(current_device.queue, gainStageMasker);

        // get the pedestal data from the device
        alpakaCopy(current_device.queue,
                   gainStage,
                   current_device.gainStageOutput,
                   SINGLEMAP);

        // wait for copy to finish
        alpakaWait(current_device.queue);

        return alpakaNativePtr(gainStage);
    }

    /**
     * Fall back to initial pedestal maps.
     * @param pedestalFallback Whether to fall back on initial pedestal values
     * or not.
     */
    auto useInitialPedestals(bool pedestalFallback) -> void
    {
        DEBUG("Using initial pedestal values:", init);
        this->pedestalFallback = pedestalFallback;
    }

    /**
     * Downloads the current drift map.
     * @return drift map
     */
    auto downloadDriftMap() -> DriftMap*
    {
        DEBUG("downloading drift map...");

        // create handle for the device with the current version of the pedestal
        // maps
        auto current_device = devices[nextFree];

        // mask gain stage maps
        DriftMapKernel driftMapKernel;
        auto const driftMap(alpakaCreateKernel<TAlpaka>(
            getWorkDiv<TAlpaka>(),
            driftMapKernel,
            alpakaNativePtr(current_device.initialPedestal),
            alpakaNativePtr(current_device.pedestal),
            alpakaNativePtr(current_device.drift)));

        alpakaEnqueueKernel(current_device.queue, driftMap);

        // get the pedestal data from the device
        alpakaCopy(
            current_device.queue, drift, current_device.drift, SINGLEMAP);

        // wait for copy to finish
        alpakaWait(current_device.queue);

        return alpakaNativePtr(drift);
    }

    /**
     * Tries to upload one data package.
     * @param Maps-struct with raw data, offset within the package
     * @return number of frames uploaded from the package
     */
    auto uploadData(FramePackage<DetectorData, TAlpaka> data,
                    std::size_t offset,
                    ExecutionFlags flags,
                    bool flushWhenFinished = true) -> std::size_t
    {
        if (!ringbuffer.isEmpty()) {
            // try uploading one data package
            if (offset <= data.numFrames - DEV_FRAMES) {
                offset += calcData(
                    alpakaNativePtr(data.data) + offset, DEV_FRAMES, flags);
                DEBUG(offset, "/", data.numFrames, "frames uploaded");
            }
            // upload remaining frames
            else if (offset != data.numFrames) {
                offset += calcData(alpakaNativePtr(data.data) + offset,
                                   data.numFrames % DEV_FRAMES,
                                   flags);
                DEBUG(offset, "/", data.numFrames, "frames uploaded");
            }
            // force wait for one device to finish since there's no new data and
            // the user wants the data flushed
            else if (flushWhenFinished) {
                flush();
            }
        }

        return offset;
    }

    /**
     * Tries to download one data package.
     * @param pointer to empty struct for photon and sum maps and cluster data
     * @return boolean indicating whether maps were downloaded or not
     */
    auto
    downloadData(boost::optional<FramePackage<EnergyMap, TAlpaka>&> energy,
                 boost::optional<FramePackage<PhotonMap, TAlpaka>&> photon,
                 boost::optional<FramePackage<SumMap, TAlpaka>&> sum,
                 boost::optional<FramePackage<EnergyValue, TAlpaka>&> maxValues,
                 boost::optional<ClusterArray<TAlpaka>&> clusters) -> size_t
    {
        struct DeviceData<TAlpaka>* dev = &Dispenser::devices[nextFree];


        DEBUG(
            "downloading from device", nextFree, "(nextFull is", nextFull, ")");
        std::string s = "";
        for (const auto& d : devices) {
            s += (d.state == READY ? "READY" : "");
            s += (d.state == FREE ? "FREE" : "");
            s += (d.state == PROCESSING ? "PROCESSING" : "");
            s += " ";
        }
        DEBUG("Current state: ", s);


        // to keep frames in order only download if the longest running device
        // has finished
        if (dev->state != READY)
            return 0;

        // download energy if needed
        if (energy) {
            DEBUG("downloading energy");
            (*energy).numFrames = dev->numMaps;
            alpakaCopy(dev->queue, (*energy).data, dev->energy, dev->numMaps);
        }

        // download photon data if needed
        if (photon) {
            DEBUG("downloading photons");
            (*photon).numFrames = dev->numMaps;
            alpakaCopy(dev->queue, (*photon).data, dev->photon, dev->numMaps);
        }

        // download summation frames if needed
        if (sum) {
            DEBUG("downloading sum");
            (*sum).numFrames = dev->numMaps / SUM_FRAMES;
            alpakaCopy(
                dev->queue, (*sum).data, dev->sum, (dev->numMaps / SUM_FRAMES));
        }

        // download maximum values if needed
        if (maxValues) {
            DEBUG("downloading max values");
            (*maxValues).numFrames = dev->numMaps;
            alpakaCopy(
                dev->queue, (*maxValues).data, dev->maxValues, DEV_FRAMES);
        }

        // download number of clusters if needed
        if (clusters) {
            DEBUG("downloading clusters");
            alpakaCopy(dev->queue,
                       (*clusters).usedPinned,
                       dev->numClusters,
                       SINGLEMAP);

            // wait for completion of copy operations
            alpakaWait(dev->queue);

            // reserve space in the cluster array for the new data and save the
            // number of clusters to download temporarily
            auto oldNumClusters = (*clusters).used;
            auto clustersToDownload =
                alpakaNativePtr((*clusters).usedPinned)[0];
            alpakaNativePtr((*clusters).usedPinned)[0] += oldNumClusters;
            (*clusters).used = alpakaNativePtr((*clusters).usedPinned)[0];

            DEBUG("clustersToDownload:", clustersToDownload);

            // create a subview in the cluster buffer where the new data shuld
            // be downloaded to
            auto const extentView(Vec(static_cast<Size>(clustersToDownload)));
            auto const offsetView(Vec(static_cast<Size>(oldNumClusters)));
            typename TAlpaka::HostView<Cluster> clusterView(
                (*clusters).clusters, extentView, offsetView);

            // download actual clusters
            alpakaCopy(
                dev->queue, clusterView, dev->cluster, clustersToDownload);
        }

        alpakaWait(dev->queue);

        dev->state = FREE;
        nextFree = (nextFree + 1) % devices.size();
        ringbuffer.push(dev);

        return dev->numMaps;
    }

    /**
     * Returns the a vector with the amount of memory of each device.
     * @return size_array
     */
    auto getMemSize() -> std::vector<std::size_t>
    {
        std::vector<std::size_t> sizes(devices.size());
        for (std::size_t i = 0; i < devices.size(); ++i) {
            sizes[i] = alpakaGetMemBytes(devices[i].device);
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
            sizes[i] = alpakaGetFreeMemBytes(devices[i].device);
        }

        return sizes;
    }

private:
    typename TAlpaka::DevHost host;
    FramePackage<GainMap, TAlpaka> gain;
    typename TAlpaka::HostBuf<MaskMap> mask;
    typename TAlpaka::HostBuf<DriftMap> drift;
    typename TAlpaka::HostBuf<GainStageMap> gainStage;
    typename TAlpaka::HostBuf<EnergyMap> maxValueMaps;

    FramePackage<PedestalMap, TAlpaka> pedestal;
    FramePackage<InitPedestalMap, TAlpaka> initPedestal;

    bool init;
    bool pedestalFallback;
    Ringbuffer<DeviceData<TAlpaka>*> ringbuffer;
    std::vector<DeviceData<TAlpaka>> devices;

    std::size_t nextFree, nextFull;

    /**
     * Initializes all devices. Uploads gain data and creates buffer.
     * @param vector with devices to be initialized
     */
    auto initDevices(std::vector<typename TAlpaka::DevAcc> allDevices) -> void
    {
        const GainmapInversionKernel gainmapInversionKernel;
        devices.reserve(allDevices.size() * TAlpaka::STREAMS_PER_DEV);
        for (std::size_t num = 0;
             num < allDevices.size() * TAlpaka::STREAMS_PER_DEV;
             ++num) {
            // initialize variables
            devices.emplace_back(num,
                                 allDevices[num / TAlpaka::STREAMS_PER_DEV]);
            alpakaCopy(
                devices[num].queue, devices[num].gain, gain.data, GAINMAPS);
            // compute reciprocals of gain maps
            auto const gainmapInversion(alpakaCreateKernel<TAlpaka>(
                getWorkDiv<TAlpaka>(),
                gainmapInversionKernel,
                alpakaNativePtr(devices[num].gain)));
            alpakaEnqueueKernel(devices[num].queue, gainmapInversion);

            if (!ringbuffer.push(&devices[num])) {
                fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
                exit(EXIT_FAILURE);
            }
            DEBUG("Device #", (num + 1), "initialized!");
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
        DeviceData<TAlpaka>* dev;
        if (!ringbuffer.pop(dev))
            return 0;

        DEBUG("calculate pedestal data on device", dev->id);

        dev->state = PROCESSING;
        dev->numMaps = numMaps;

        alpakaCopy(
            dev->queue,
            dev->data,
            alpakaViewPlainPtrHost<TAlpaka, TDetectorData>(data, host, numMaps),
            numMaps);

        // copy offset data from last initialized device
        auto prevDevice = (nextFull + devices.size() - 1) % devices.size();
        alpakaWait(devices[prevDevice].queue);
        alpakaCopy(
            dev->queue, dev->pedestal, devices[prevDevice].pedestal, PEDEMAPS);
        alpakaCopy(dev->queue,
                   dev->initialPedestal,
                   devices[prevDevice].initialPedestal,
                   PEDEMAPS);

        alpakaCopy(dev->queue, dev->mask, devices[prevDevice].mask, SINGLEMAP);

        // increase nextFull and nextFree (because pedestal data isn't
        // downloaded like normal data)
        nextFull = (nextFull + 1) % devices.size();
        nextFree = (nextFree + 1) % devices.size();

        if (!init) {
            alpakaMemSet(dev->queue, dev->pedestal, 0, SINGLEMAP);
            alpakaWait(dev->queue);
            init = true;
        }

        CalibrationKernel calibrationKernel;
        auto const calibration(
            alpakaCreateKernel<TAlpaka>(getWorkDiv<TAlpaka>(),
                                        calibrationKernel,
                                        alpakaNativePtr(dev->data),
                                        alpakaNativePtr(dev->initialPedestal),
                                        alpakaNativePtr(dev->pedestal),
                                        alpakaNativePtr(dev->mask),
                                        dev->numMaps));

        alpakaEnqueueKernel(dev->queue, calibration);

        alpakaWait(dev->queue);

        dev->state = FREE;

        if (!ringbuffer.push(dev)) {
            fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
            exit(EXIT_FAILURE);
        }

        return numMaps;
    }

    /**
     * Masks all pixels over a certain standard deviation threshold.
     * @param threshold standard deviation threshold.
     */
    auto maskStdDevOver(double threshold) -> void
    {
        DEBUG("checking stddev (on device", nextFull, ")");

        // create stddev check kernel object
        CheckStdDevKernel checkStdDevKernel;
        auto const checkStdDev(alpakaCreateKernel<TAlpaka>(
            getWorkDiv<TAlpaka>(),
            checkStdDevKernel,
            alpakaNativePtr(devices[nextFull].initialPedestal),
            alpakaNativePtr(devices[nextFull].mask),
            threshold));

        alpakaEnqueueKernel(devices[nextFull].queue, checkStdDev);
    }

    /**
     * Distributes copies the mask map of the current accelerator to all others.
     */
    auto distributeMaskMaps() -> void
    {
        uint64_t source = (nextFull + devices.size() - 1) % devices.size();
        DEBUG("distributeMaskMaps (from", source, ")");
        for (uint64_t i = 0; i < devices.size() - 1; ++i) {
            uint64_t destination =
                (i + source + (i >= source ? 1 : 0)) % devices.size();
            alpakaCopy(devices[source].queue,
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
        DEBUG("distribute initial pedestal maps (from", source, ")");
        for (uint64_t i = 0; i < devices.size(); ++i) {
            // distribute initial pedestal map (containing statistics etc.)
            alpakaCopy(devices[source].queue,
                       devices[i].initialPedestal,
                       devices[source].initialPedestal,
                       SINGLEMAP);

            // distribute pedestal map (with initial data)
            alpakaCopy(devices[source].queue,
                       devices[i].pedestal,
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
        DeviceData<TAlpaka>* dev;
        if (!ringbuffer.pop(dev))
            return 0;

        dev->state = PROCESSING;
        dev->numMaps = numMaps;

        alpakaCopy(
            dev->queue,
            dev->data,
            alpakaViewPlainPtrHost<TAlpaka, DetectorData>(data, host, numMaps),
            numMaps);

        // copy offset data from last device uploaded to
        auto prevDevice = (nextFull + devices.size() - 1) % devices.size();
        alpakaWait(dev->queue, devices[prevDevice].event);
        DEBUG("device", devices[prevDevice].id, "finished");

        devices[prevDevice].state = READY;
        alpakaCopy(
            dev->queue, dev->pedestal, devices[prevDevice].pedestal, PEDEMAPS);

        nextFull = (nextFull + 1) % devices.size();

        // reset the number of clusters
        alpakaMemSet(dev->queue, dev->numClusters, 0, SINGLEMAP);

        MaskMap* local_mask =
            flags.masking ? alpakaNativePtr(dev->mask) : nullptr;

        // converting to energy
        // the photon and cluster extraction kernels already include energy
        // conversion
        if (flags.mode == 0) {
            ConversionKernel conversionKernel;
            auto const conversion(alpakaCreateKernel<TAlpaka>(
                getWorkDiv<TAlpaka>(),
                conversionKernel,
                alpakaNativePtr(dev->data),
                alpakaNativePtr(dev->gain),
                alpakaNativePtr(dev->initialPedestal),
                alpakaNativePtr(dev->pedestal),
                alpakaNativePtr(dev->gainStage),
                alpakaNativePtr(dev->energy),
                dev->numMaps,
                local_mask,
                pedestalFallback));

            DEBUG("enqueueing conversion kernel");
            alpakaEnqueueKernel(dev->queue, conversion);
        }

        // converting to photons (and energy)
        if (flags.mode == 1) {
            PhotonFinderKernel photonFinderKernel;
            auto const photonFinder(alpakaCreateKernel<TAlpaka>(
                getWorkDiv<TAlpaka>(),
                photonFinderKernel,
                alpakaNativePtr(dev->data),
                alpakaNativePtr(dev->gain),
                alpakaNativePtr(dev->initialPedestal),
                alpakaNativePtr(dev->pedestal),
                alpakaNativePtr(dev->gainStage),
                alpakaNativePtr(dev->energy),
                alpakaNativePtr(dev->photon),
                dev->numMaps,
                local_mask,
                pedestalFallback));

            DEBUG("enqueueing photon kernel");
            alpakaEnqueueKernel(dev->queue, photonFinder);
        };

        // clustering (and conversion to energy)
        if (flags.mode == 2) {
            DEBUG("enqueueing clustering kernel");

            for (uint32_t i = 0; i < numMaps + 1; ++i) {
                // execute the clusterfinder with the pedestalupdate on every
                // frame
                ClusterFinderKernel clusterFinderKernel;
                auto const clusterFinder(alpakaCreateKernel<TAlpaka>(
                    getWorkDiv<TAlpaka>(),
                    clusterFinderKernel,
                    alpakaNativePtr(dev->data),
                    alpakaNativePtr(dev->gain),
                    alpakaNativePtr(dev->initialPedestal),
                    alpakaNativePtr(dev->pedestal),
                    alpakaNativePtr(dev->gainStage),
                    alpakaNativePtr(dev->energy),
                    alpakaNativePtr(dev->cluster),
                    alpakaNativePtr(dev->numClusters),
                    local_mask,
                    dev->numMaps,
                    i,
                    pedestalFallback));

                alpakaEnqueueKernel(dev->queue, clusterFinder);
            }
        }

        // find max value
        if (flags.maxValue) {
            // get the max value
            for (uint32_t i = 0; i < numMaps; ++i) {
                // reduce all images
                WorkDiv workdivRun1(
                    decltype(TAlpaka::blocksPerGrid)(TAlpaka::blocksPerGrid),
                    decltype(TAlpaka::threadsPerBlock)(
                        TAlpaka::threadsPerBlock),
                    static_cast<Size>(1));
                ReduceKernel<TAlpaka::threadsPerBlock, double> reduceKernelRun1;
                auto const reduceRun1(alpakaCreateKernel<TAlpaka>(
                    workdivRun1,
                    reduceKernelRun1,
                    &alpakaNativePtr(dev->energy)[i],
                    &alpakaNativePtr(dev->maxValueMaps)[i],
                    DIMX * DIMY));

                WorkDiv workdivRun2{static_cast<Size>(1),
                                    decltype(TAlpaka::threadsPerBlock)(
                                        TAlpaka::threadsPerBlock),
                                    static_cast<Size>(1)};
                ReduceKernel<TAlpaka::threadsPerBlock, double> reduceKernelRun2;
                auto const reduceRun2(alpakaCreateKernel<TAlpaka>(
                    workdivRun2,
                    reduceKernelRun2,
                    &alpakaNativePtr(dev->maxValueMaps)[i],
                    &alpakaNativePtr(dev->maxValueMaps)[i],
                    decltype(TAlpaka::blocksPerGrid)(TAlpaka::blocksPerGrid)));
                alpakaEnqueueKernel(dev->queue, reduceRun1);
                alpakaEnqueueKernel(dev->queue, reduceRun2);
            }

            WorkDiv workdivMaxValueCopy{
                static_cast<Size>(std::ceil((double)numMaps /
                                            decltype(TAlpaka::threadsPerBlock)(
                                                TAlpaka::threadsPerBlock))),
                decltype(TAlpaka::threadsPerBlock)(TAlpaka::threadsPerBlock),
                static_cast<Size>(1)};
            MaxValueCopyKernel maxValueCopyKernel;
            auto const maxValueCopy(
                alpakaCreateKernel<TAlpaka>(workdivMaxValueCopy,
                                            maxValueCopyKernel,
                                            alpakaNativePtr(dev->maxValueMaps),
                                            alpakaNativePtr(dev->maxValues),
                                            numMaps));

            DEBUG("enqueueing max value extraction kernel");
            alpakaEnqueueKernel(dev->queue, maxValueCopy);
        }

        // summation
        if (flags.summation) {
            DEBUG("enqueueing summation kernel");

            SummationKernel summationKernel;
            auto const summation(
                alpakaCreateKernel<TAlpaka>(getWorkDiv<TAlpaka>(),
                                            summationKernel,
                                            alpakaNativePtr(dev->energy),
                                            dev->numMaps,
                                            SUM_FRAMES,
                                            alpakaNativePtr(dev->sum)));

            alpakaEnqueueKernel(dev->queue, summation);
        };

        // the event is used to wait for pedestal data
        alpakaEnqueueKernel(dev->queue, dev->event);

        return numMaps;
    }
};
