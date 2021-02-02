#pragma once

#include "Alpakaconfig.hpp"
#include "Config.hpp"
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

#include <optional.hpp>

#include <future>
#include <iostream>
#include <limits>

template <typename TConfig, template <std::size_t> typename TAccelerator>
class Dispenser {
public:
  // use types defined in the config struct
  using TAlpaka = TAccelerator<TConfig::MAPSIZE>;
  using MaskMap = typename TConfig::MaskMap;

  /**
   * Dispenser constructor
   * @param Maps-Struct with initial gain
   */
  Dispenser(FramePackage<typename TConfig::GainMap, TAlpaka> gainMap,
            double beamConst,
            tl::optional<typename TAlpaka::template HostBuf<MaskMap>> mask,
            unsigned int moduleNumber = 0, unsigned int moduleCount = 1)
      : gain(gainMap),
        mask((mask ? *mask
                   : alpakaAlloc<typename TConfig::MaskMap>(
                         alpakaGetHost<TAlpaka>(),
                         decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP)))),
        drift(alpakaAlloc<typename TConfig::DriftMap>(
            alpakaGetHost<TAlpaka>(),
            decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP))),
        gainStage(decltype(TConfig::DEV_FRAMES)(TConfig::DEV_FRAMES)),
        maxValueMaps(alpakaAlloc<typename TConfig::EnergyMap>(
            alpakaGetHost<TAlpaka>(),
            decltype(TConfig::DEV_FRAMES)(TConfig::DEV_FRAMES))),
        pedestalFallback(false), init(false),
        ringbuffer(TAlpaka::STREAMS_PER_DEV * alpakaGetDevCount<TAlpaka>()),
        pedestal(TConfig::PEDEMAPS, alpakaGetHost<TAlpaka>()),
        initPedestal(TConfig::PEDEMAPS, alpakaGetHost<TAlpaka>()),
        beamConst(beamConst), nextFull(0), nextFree(0),
        deviceContainer(alpakaGetDevs<TAlpaka>()), moduleNumber(moduleNumber),
        moduleCount(moduleCount) {
    initDevices();

    // make room for live mask information
    if (!mask) {
      alpakaMemSet(devices[0].queue, devices[0].mask, 1,
                   decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));
    }

    synchronize();
  }

  /**
   * copy constructor deleted
   */
  Dispenser(const Dispenser &other) = delete;

  /**
   * assign constructor deleted
   */
  Dispenser &operator=(const Dispenser &other) = delete;

  /**
   * move copy constructor
   */
  Dispenser(Dispenser &&other) = default;

  /**
   * move assign operator
   */
  Dispenser &operator=(Dispenser &&other) = default;

  /**
   * Synchronizes all streams with one function call.
   */
  auto synchronize() -> void {
    DEBUG("synchronizing devices ...");

    for (struct DeviceData<TConfig, TAlpaka> &dev : devices)
      alpakaWait(dev.queue);
  }

  auto reset() -> void {
    // reset variables
    pedestalFallback = false;
    init = false;
    nextFull = 0;
    nextFree = 0;
    ringbuffer.reset();
    devices.clear();

    // init devices
    initDevices();

    // clear mask
    alpakaMemSet(devices[0].queue, devices[0].mask, 1,
                 decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

    // synchronize
    synchronize();
  }

  /**
   * Tries to upload all data packages requiered for the init offset.
   * Only stops after all data packages are uploaded.
   * @param Maps-Struct with datamaps
   * @param stdDevThreshold An standard deviation threshold above which pixels
   * should be masked. If this is 0, no pixels will be masked.
   */
  auto
  uploadPedestaldata(FramePackage<typename TConfig::DetectorData, TAlpaka> data,
                     double stdDevThreshold = 0) -> void {
    std::size_t offset = 0;
    DEBUG("uploading pedestaldata...");

    // upload all frames cut into smaller packages
    while (offset <= data.numFrames - TConfig::DEV_FRAMES) {
      offset += calcPedestaldata(alpakaNativePtr(data.data) + offset,
                                 TConfig::DEV_FRAMES);
    }

    // upload remaining frames
    if (offset != data.numFrames) {
      offset += calcPedestaldata(alpakaNativePtr(data.data) + offset,
                                 data.numFrames % TConfig::DEV_FRAMES);
    }

    // masked values over a certain threshold if this feature is enabled
    if (stdDevThreshold != 0)
      maskStdDevOver(stdDevThreshold);

    // distribute the generated mask map and the initially generated
    // pedestal map from the current device to all others
    distributeMaskMaps();
    distributeInitialPedestalMaps();

    // signal that no clusters need to be downloaded
    for (auto &d : devices)
      d.clusterDownload = std::async(std::launch::async, []() {});
  }

  /**
   * Downloads the pedestal data.
   * @return pedestal pedestal data
   */
  auto downloadPedestaldata()
      -> FramePackage<typename TConfig::PedestalMap, TAlpaka> {
    //! @todo: find a more beautiful solution than this
    // get current device number
    uint64_t source = (nextFull + devices.size() - 1) % devices.size();

    // create handle for the device with the current version of the pedestal
    // maps
    auto &current_device = devices[source];

    DEBUG("downloading pedestaldata from device", source);

    // get the pedestal data from the device
    alpakaCopy(current_device.queue, pedestal.data, current_device.pedestal,
               decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

    // wait for copy to finish
    alpakaWait(current_device.queue);

    pedestal.numFrames = TConfig::PEDEMAPS;

    return pedestal;
  }

  /**
   * Downloads the initial pedestal data.
   * @return pedestal pedestal data
   */
  auto downloadInitialPedestaldata()
      -> FramePackage<typename TConfig::InitPedestalMap, TAlpaka> {
    //! @todo: find a more beautifil solution than this
    // get current device number
    uint64_t source = (nextFull + devices.size() - 1) % devices.size();

    // create handle for the device with the current version of the pedestal
    // maps
    auto &current_device = devices[source];
    DEBUG("downloading pedestaldata from device", source);

    // get the pedestal data from the device
    alpakaCopy(current_device.queue, initPedestal.data,
               current_device.initialPedestal,
               decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

    // wait for copy to finish
    alpakaWait(current_device.queue);

    initPedestal.numFrames = TConfig::PEDEMAPS;

    return initPedestal;
  }

  /**
   * Downloads the current mask map.
   * @return mask map
   */
  auto downloadMask() -> typename TConfig::MaskMap * {
    DEBUG("downloading mask...");

    //! @todo: find a more beautifil solution than this
    // get current device number
    uint64_t source = (nextFull + devices.size() - 1) % devices.size();

    // create handle for the device with the current version of the pedestal
    // maps
    auto &current_device = devices[source];

    // get the pedestal data from the device
    alpakaCopy(current_device.queue, mask, current_device.mask,
               decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

    // wait for copy to finish
    alpakaWait(current_device.queue);

    return alpakaNativePtr(mask);
  }

  /**
   * Flags alldevices as ready for download.
   */
  auto flush() -> void {
    DEBUG("flushing...");
    synchronize();
    for (auto &device : devices)
      if (device.state != FREE)
        device.state = READY;
  }

  /**
   * Downloads the current gain stage map.
   * @return gain stage map
   */
  auto downloadGainStages()
      -> FramePackage<typename TConfig::GainStageMap, TAlpaka> {
    DEBUG("downloading gain stage map...");

    //! @todo: find a more beautifil solution than this
    // get current device number
    uint64_t source = (nextFull + devices.size() - 1) % devices.size();

    // create handle for the device with the current version of the pedestal
    // maps
    auto &current_device = devices[source];

    // mask gain stage maps
    GainStageMaskingKernel<TConfig> gainStageMasking;
    auto const gainStageMasker(alpakaCreateKernel<TAlpaka>(
        getWorkDiv<TAlpaka>(), gainStageMasking,
        alpakaNativePtr(current_device.gainStage),
        alpakaNativePtr(current_device.gainStageOutput), current_device.numMaps,
        alpakaNativePtr(current_device.mask)));

    alpakaEnqueueKernel(current_device.queue, gainStageMasker);

    // get the pedestal data from the device
    alpakaCopy(current_device.queue, gainStage.data,
               current_device.gainStageOutput, current_device.numMaps);

    // wait for copy to finish
    alpakaWait(current_device.queue);

    gainStage.numFrames = current_device.numMaps;

    return gainStage;
  }

  /**
   * Fall back to initial pedestal maps.
   * @param pedestalFallback Whether to fall back on initial pedestal values
   * or not.
   */
  auto useInitialPedestals(bool pedestalFallback) -> void {
    DEBUG("Using initial pedestal values:", init);
    this->pedestalFallback = pedestalFallback;
  }

  /**
   * Downloads the current drift map.
   * @return drift map
   */
  auto downloadDriftMap() -> typename TConfig::DriftMap * {
    DEBUG("downloading drift map...");

    //! @todo: find a more beautifil solution than this
    // get current device number
    uint64_t source = (nextFull + devices.size() - 1) % devices.size();

    // create handle for the device with the current version of the pedestal
    // maps
    auto &current_device = devices[source];

    // mask gain stage maps
    DriftMapKernel<TConfig> driftMapKernel;
    auto const driftMap(alpakaCreateKernel<TAlpaka>(
        getWorkDiv<TAlpaka>(), driftMapKernel,
        alpakaNativePtr(current_device.initialPedestal),
        alpakaNativePtr(current_device.pedestal),
        alpakaNativePtr(current_device.drift)));

    alpakaEnqueueKernel(current_device.queue, driftMap);

    // get the pedestal data from the device
    alpakaCopy(current_device.queue, drift, current_device.drift,
               decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

    // wait for copy to finish
    alpakaWait(current_device.queue);

    return alpakaNativePtr(drift);
  }

  template <typename TFramePackageEnergyMap, typename TFramePackagePhotonMap,
            typename TFramePackageSumMap, typename TFramePackageEnergyValue>
  auto process(FramePackage<typename TConfig::DetectorData, TAlpaka> data,
               std::size_t offset, ExecutionFlags flags,
               tl::optional<TFramePackageEnergyMap> energy,
               tl::optional<TFramePackagePhotonMap> photon,
               tl::optional<TFramePackageSumMap> sum,
               tl::optional<TFramePackageEnergyValue> maxValues,
               typename TConfig::template ClusterArray<TAlpaka> *clusters,
               bool flushWhenFinished = true)
      -> std::tuple<std::size_t, std::future<bool>> {

    uint32_t uploadOffset = offset;
    // try uploading a data package to every device
    for (uint32_t i = 0; i < devices.size() && uploadOffset < data.numFrames;
         ++i) {

      // create data view
      uint32_t framesToProcess =
          std::min(TConfig::DEV_FRAMES, data.numFrames - uploadOffset);
      auto dataView = data.getView(uploadOffset, framesToProcess);

      // get current device
      uint32_t current_device = (nextFree + i) % devices.size();
      DeviceData<TConfig, TAlpaka> *dev = &devices[current_device];

      DEBUG("Uploading", framesToProcess, "frames to device", current_device);

      // create a shadow buffer on the device and upload data if required
      FramePackageDoubleBuffer<TAlpaka, decltype(dataView), decltype(dev->data)>
          dataBuffer(dataView, dev->data, &dev->queue, framesToProcess);
      dataBuffer.upload();

      uploadOffset += framesToProcess;
    }

    // initialize offsets
    uint64_t sum_offset = 0;
    uint64_t view_offset = 0;
    uint64_t max_view_offset = 0;
    if (energy)
      max_view_offset = energy->numFrames;
    else if (photon)
      max_view_offset = photon->numFrames;

    // create vector for futures
    std::vector<std::tuple<std::size_t, std::future<bool>>> results;

    // start processing
    for (uint32_t i = 0; i < devices.size() && offset < data.numFrames &&
                         view_offset < max_view_offset;
         ++i) {
      // create data view
      uint32_t framesToProcess =
          std::min(TConfig::DEV_FRAMES, data.numFrames - offset);
      uint64_t sumFrames =
          (framesToProcess + TConfig::SUM_FRAMES - 1) / TConfig::SUM_FRAMES;

      // define views
      auto dataView = data.getView(offset, framesToProcess);
      auto energy_view([&]() -> tl::optional<TFramePackageEnergyMap> {
        if (energy)
          return energy->getView(view_offset, framesToProcess);
        return tl::nullopt;
      }());
      auto photon_view([&]() -> tl::optional<TFramePackagePhotonMap> {
        if (photon)
          return photon->getView(view_offset, framesToProcess);
        return tl::nullopt;
      }());
      auto sum_view([&]() -> tl::optional<TFramePackageSumMap> {
        if (sum)
          return sum->getView(sum_offset, sumFrames);
        return tl::nullopt;
      }());
      auto maxValues_view([&]() -> tl::optional<TFramePackageEnergyValue> {
        if (maxValues)
          return maxValues->getView(view_offset, framesToProcess);
        return tl::nullopt;
      }());

      DEBUG("Processing", framesToProcess, "frames");

      // process data
      results.emplace_back(processData(dataView, framesToProcess, flags,
                                       energy_view, photon_view, sum_view,
                                       maxValues_view, clusters));

      // update offsets
      uint64_t new_offset = std::get<0>(*results.rbegin());
      offset += new_offset;
      view_offset += new_offset;
      sum_offset +=
          (new_offset + TConfig::SUM_FRAMES - 1) / TConfig::SUM_FRAMES;
      DEBUG(offset, "/", data.numFrames, "frames processed");
    }

    // force wait for one device to finish since there's no new data and
    // the user wants the data flushed
    if (flushWhenFinished && offset >= data.numFrames) {
      DEBUG("flushing ...");

      flush();
    }

    // work around CUDA bug that doesn't let me move capture objects
    auto shared_results = std::make_shared<
        std::vector<std::tuple<std::size_t, std::future<bool>>>>(
        std::move(results));

    return std::make_tuple(offset,
                           std::async(std::launch::async, [shared_results]() {
                             shared_results->clear();
                             return true;
                           }));
  }

  /**
   * Returns the a vector with the amount of memory of each device.
   * @return size_array
   */
  auto getMemSize() -> std::vector<std::size_t> {
    std::vector<std::size_t> sizes(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
      sizes[i] = alpakaGetMemBytes(*devices[i].device);
    }

    return sizes;
  }

  /**
   * Returns the a vector with the amount of free memory of each device.
   * @return size_array
   */
  auto getFreeMem() -> std::vector<std::size_t> {
    std::vector<std::size_t> sizes(devices.size());
    for (std::size_t i = 0; i < devices.size(); ++i) {
      sizes[i] = alpakaGetFreeMemBytes(*devices[i].device);
    }

    return sizes;
  }

private:
  FramePackage<typename TConfig::GainMap, TAlpaka> gain;
  typename TAlpaka::template HostBuf<typename TConfig::MaskMap> mask;
  typename TAlpaka::template HostBuf<typename TConfig::DriftMap> drift;
  FramePackage<typename TConfig::GainStageMap, TAlpaka> gainStage;
  typename TAlpaka::template HostBuf<typename TConfig::EnergyMap> maxValueMaps;

  FramePackage<typename TConfig::PedestalMap, TAlpaka> pedestal;
  FramePackage<typename TConfig::InitPedestalMap, TAlpaka> initPedestal;

  std::vector<typename TAlpaka::DevAcc> deviceContainer;

  bool init;
  bool pedestalFallback;
  Ringbuffer<DeviceData<TConfig, TAlpaka> *> ringbuffer;
  std::vector<DeviceData<TConfig, TAlpaka>> devices;

  double beamConst;

  std::size_t nextFree, nextFull;
  unsigned int moduleNumber, moduleCount;

  /**
   * Initializes all devices. Uploads gain data and creates buffer.
   * @param vector with devices to be initialized
   */
  auto initDevices() -> void {

    DEBUG("Starting device initialization");

    const GainmapInversionKernel<TConfig> gainmapInversionKernel{};
    std::size_t maxQueues =
        alpakaGetDevCount<TAlpaka>() * TAlpaka::STREAMS_PER_DEV;
    std::size_t deviceCount = static_cast<unsigned int>(std::ceil(
        static_cast<double>(maxQueues) / static_cast<double>(moduleCount)));

    //! @todo: find all the other debug code
    devices.reserve(deviceCount);

    for (std::size_t num = 0; num < deviceCount; ++num) {

      DEBUG("Initializing device #", num);

      // initialize variables
      std::size_t selectedQueue = (num + moduleNumber * deviceCount);

      /* // performs poorly on GPUs
      DEBUG("Initializing queue", selectedQueue, "on device",
            selectedQueue % ((alpakaGetDevCount<TAlpaka>() + moduleCount - 1) /
                             moduleCount));

      devices.emplace_back(
          selectedQueue,
          &deviceContainer[selectedQueue %
                           ((alpakaGetDevCount<TAlpaka>() + moduleCount - 1) /
                            moduleCount)]);
      */

      DEBUG("Initializing queue", selectedQueue, "on device",
            selectedQueue / TAlpaka::STREAMS_PER_DEV);

      devices.emplace_back(
          selectedQueue,
          &deviceContainer[selectedQueue / TAlpaka::STREAMS_PER_DEV]);

      // init cluster data to 0
      alpakaMemSet(devices[num].queue, devices[num].cluster, 0,
                   decltype(TConfig::MAX_CLUSTER_NUM_USER)(
                       TConfig::MAX_CLUSTER_NUM_USER) *
                       decltype(TConfig::DEV_FRAMES)(TConfig::DEV_FRAMES));

      // copy gain maps to device
      alpakaCopy(devices[num].queue, devices[num].gain, gain.data,
                 decltype(TConfig::GAINMAPS)(TConfig::GAINMAPS));

      // compute reciprocals of gain maps
      auto const gainmapInversion(alpakaCreateKernel<TAlpaka>(
          getWorkDiv<TAlpaka>(), gainmapInversionKernel,
          alpakaNativePtr(devices[num].gain)));
      alpakaEnqueueKernel(devices[num].queue, gainmapInversion);

      // place the initialized device into a ringbuffer
      if (!ringbuffer.push(&devices[num])) {
        fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
        exit(EXIT_FAILURE);
      }
      DEBUG("Device #", (num + moduleNumber * deviceCount + 1), "initialized!");
    }
  }

  /**
   * Executes the calibration kernel.
   * @param pointer to raw data and number of frames
   * @return number of frames calculated
   */
  template <typename TDetectorData>
  auto calcPedestaldata(TDetectorData *data, std::size_t numMaps)
      -> std::size_t {
    // get the next free device from the ringbuffer
    DeviceData<TConfig, TAlpaka> *dev;
    if (!ringbuffer.pop(dev))
      return 0;

    DEBUG("calculate pedestal data on device", dev->id);

    // set the state to processing
    dev->state = PROCESSING;
    dev->numMaps = numMaps;

    // upload the data to the device
    alpakaCopy(dev->queue, dev->data,
               alpakaViewPlainPtrHost<TAlpaka, TDetectorData>(
                   data, alpakaGetHost<TAlpaka>(), numMaps),
               numMaps);

    // copy offset data from last initialized device (if needed)
    if (init) {
      auto prevDevice = (nextFull + devices.size() - 1) % devices.size();
      alpakaWait(devices[prevDevice].queue);

      alpakaCopy(dev->queue, dev->pedestal, devices[prevDevice].pedestal,
                 decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

      alpakaCopy(dev->queue, dev->initialPedestal,
                 devices[prevDevice].initialPedestal,
                 decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

      alpakaCopy(dev->queue, dev->mask, devices[prevDevice].mask,
                 decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));
    }

    // increase nextFull and nextFree (because pedestal data isn't
    // downloaded like normal data)
    nextFull = (nextFull + 1) % devices.size();
    nextFree = (nextFree + 1) % devices.size();

    // zero out initial pedestal maps and normal pedestal maps
    if (!init) {
      alpakaMemSet(dev->queue, dev->pedestal, 0,
                   decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));
      alpakaMemSet(dev->queue, dev->initialPedestal, 0,
                   decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));
      alpakaWait(dev->queue);
      init = true;
    }

    // execute the calibration kernel
    CalibrationKernel<TConfig> calibrationKernel{};
    auto const calibration(alpakaCreateKernel<TAlpaka>(
        getWorkDiv<TAlpaka>(), calibrationKernel, alpakaNativePtr(dev->data),
        alpakaNativePtr(dev->initialPedestal), alpakaNativePtr(dev->pedestal),
        alpakaNativePtr(dev->mask), dev->numMaps));

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
  auto maskStdDevOver(double threshold) -> void {
    DEBUG("checking stddev (on device", nextFull, ")");

    // create stddev check kernel object
    CheckStdDevKernel<TConfig> checkStdDevKernel{};
    auto const checkStdDev(alpakaCreateKernel<TAlpaka>(
        getWorkDiv<TAlpaka>(), checkStdDevKernel,
        alpakaNativePtr(devices[nextFull].initialPedestal),
        alpakaNativePtr(devices[nextFull].mask), threshold));

    alpakaEnqueueKernel(devices[nextFull].queue, checkStdDev);
  }

  /**
   * Distributes copies the mask map of the current accelerator to all others.
   */
  auto distributeMaskMaps() -> void {
    //! @todo: merge this with the function below since this function doesn't
    //! seem to work correctly and is integrated into the function below anyway
    uint64_t source = (nextFull + devices.size() - 1) % devices.size();
    DEBUG("distributeMaskMaps (from", source, ")");
    for (uint64_t i = 0; i < devices.size() - 1; ++i) {
      uint64_t destination =
          (i + source + (i >= source ? 1 : 0)) % devices.size();
      alpakaCopy(devices[source].queue, devices[destination].mask,
                 devices[source].mask,
                 decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));
    }
    synchronize();
  }

  /**
   * Distributes copies the initial pedestal map of the current accelerator to
   * all others.
   */
  auto distributeInitialPedestalMaps() -> void {
    uint64_t source = (nextFull + devices.size() - 1) % devices.size();
    DEBUG("distribute initial pedestal maps (from", source, ")");
    for (uint64_t i = 0; i < devices.size(); ++i) {
      // distribute initial pedestal map (containing statistics etc.)
      alpakaCopy(devices[source].queue, devices[i].initialPedestal,
                 devices[source].initialPedestal,
                 decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

      // distribute pedestal map (with initial data)
      alpakaCopy(devices[source].queue, devices[i].pedestal,
                 devices[source].pedestal,
                 decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

      // distribute mask map
      alpakaCopy(devices[source].queue, devices[i].mask, devices[source].mask,
                 decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));
    }
    synchronize();
  }

  template <typename TFramePackageDetectorData, typename TFramePackageEnergyMap,
            typename TFramePackagePhotonMap, typename TFramePackageSumMap,
            typename TFramePackageEnergyValue>
  auto processData(TFramePackageDetectorData data, std::size_t numMaps,
                   ExecutionFlags flags,
                   tl::optional<TFramePackageEnergyMap> energy,
                   tl::optional<TFramePackagePhotonMap> photon,
                   tl::optional<TFramePackageSumMap> sum,
                   tl::optional<TFramePackageEnergyValue> maxValues,
                   typename TConfig::template ClusterArray<TAlpaka> *clusters)
      -> std::tuple<std::size_t, std::future<bool>> {

    using ClusterView =
        typename TAlpaka::template HostView<typename TConfig::Cluster>;

    //! @todo: pass numMaps through data.numFrames??
    DeviceData<TConfig, TAlpaka> *dev = &devices[nextFree];
    dev->numMaps = numMaps;

    // create a shadow buffer on the device and upload data if required
    FramePackageDoubleBuffer<TAlpaka, TFramePackageDetectorData,
                             decltype(dev->data)>
        dataBuffer(data, dev->data, &dev->queue, numMaps);
    // dataBuffer.upload();

    // create shadow buffers on the device if required
    FramePackageDoubleBuffer<TAlpaka, TFramePackageEnergyMap,
                             decltype(dev->energy)>
        energyBuffer(energy, dev->energy, &dev->queue, numMaps);
    FramePackageDoubleBuffer<TAlpaka, TFramePackagePhotonMap,
                             decltype(dev->photon)>
        photonBuffer(photon, dev->photon, &dev->queue, numMaps);
    FramePackageDoubleBuffer<TAlpaka, TFramePackageSumMap, decltype(dev->sum)>
        sumBuffer(sum, dev->sum, &dev->queue,
                  (numMaps + TConfig::SUM_FRAMES - 1) / TConfig::SUM_FRAMES);
    FramePackageDoubleBuffer<TAlpaka, TFramePackageEnergyValue,
                             decltype(dev->maxValues)>
        maxValuesBuffer(maxValues, dev->maxValues, &dev->queue, numMaps);

    // wait for clusters to be downloaded
    auto prevDevice = (nextFull + devices.size() - 1) % devices.size();
    if (clusters)
      devices[prevDevice].clusterDownload.wait();

    // convert pointer to tl::optional
    tl::optional<ClusterView> optionalClusters;
    tl::optional<decltype(clusters->usedPinned)> optionalNumClusters;
    if (clusters) {
      // create subview of the free part of the cluster array
      optionalClusters.emplace(clusters->clusters,
                               alpakaGetExtent<0>(clusters->clusters) -
                                   clusters->used,
                               clusters->used);
      optionalNumClusters = clusters->usedPinned;
    }

    // temporarily set clusterBuffer size to 0 because the correct number of
    // clusters to download is not yet known
    using ClusterBufferType =
        typename GetDoubleBuffer<TAlpaka, ClusterView,
                                 decltype(dev->cluster)>::Buffer;
    auto clusterBuffer = std::make_shared<ClusterBufferType>(
        optionalClusters, dev->cluster, &dev->queue,
        static_cast<std::size_t>(0));
    typename GetDoubleBuffer<TAlpaka, decltype(clusters->usedPinned),
                             decltype(dev->numClusters)>::Buffer
        clustersUsedBuffer(optionalNumClusters, dev->numClusters, &dev->queue,
                           TConfig::SINGLEMAP);

    // copy offset data from last device uploaded to current device
    alpakaWait(dev->queue, devices[prevDevice].event);
    DEBUG("device", devices[prevDevice].id, "finished");

    devices[prevDevice].state = READY;
    alpakaCopy(dev->queue, dev->pedestal, devices[prevDevice].pedestal,
               decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

    nextFull = (nextFull + 1) % devices.size();

    // enqueue the kernels
    enqueueKernels(dev, numMaps, flags, dataBuffer.get(), energyBuffer.get(),
                   photonBuffer.get(), sumBuffer.get(), maxValuesBuffer.get(),
                   clusterBuffer.get()->get(), clustersUsedBuffer.get());

    // download the data
    if (clusters) {

      // download the number of found clusters
      clustersUsedBuffer.download();

      /* TODO: unfortunately this deadlocks with the CUDA backend
      // enqueue download of actual cluster data
      alpakaEnqueueKernel(dev->queue, [dev, clusters, clusterBuffer]() {
        auto clustersToDownload = alpakaNativePtr(clusters->usedPinned)[0];
        clusters->used += clustersToDownload;

        DEBUG("Downloading ", clustersToDownload, "clusters (", clusters->used,
              "in total) from device", dev->id);

        clusterBuffer->resize(clustersToDownload);

        DEBUG("Resizing");

        clusterBuffer->download();
        DEBUG("Download finished");
      });*/

      typename TAlpaka::Event event(*dev->device);
      alpakaEnqueueKernel(dev->queue, event);

      // download clusters asynchronously
      dev->clusterDownload = std::async(
          std::launch::async,
          [dev, clusters](decltype(clusterBuffer) &&clusterBuffer,
                          typename TAlpaka::Event &&event) {
            // wait for completion of copy operations
            // record event and then wait for is (this is more efficient
            // than a synchronization since the synchronization is blocking)
            alpakaWait(event);

            auto clustersToDownload = alpakaNativePtr(clusters->usedPinned)[0];
            clusters->used += clustersToDownload;

            DEBUG("Downloading ", clustersToDownload, "clusters (",
                  clusters->used, "in total) from device", dev->id);

            clusterBuffer->resize(clustersToDownload);
            clusterBuffer->download();
          },
          clusterBuffer, event);
    }

    DEBUG("Enqueueing event");

    // the event is used to wait for pedestal data and the cluster array
    // resizing
    alpakaEnqueueKernel(dev->queue, dev->event);

    energyBuffer.download();
    photonBuffer.download();
    sumBuffer.download();
    maxValuesBuffer.download();

    // update the nextFree index
    nextFree = (nextFree + 1) % devices.size();

    auto wait = [](decltype(dev) dev) {
      alpakaWait(dev->queue);
      dev->clusterDownload.wait();
      return true;
    };

    //! @todo: no need to return the numMaps here
    return std::make_tuple(numMaps,
                           std::async(std::launch::deferred, wait, dev));
  }

  template <typename TData, typename TOptionalEnergy, typename TOptionalPhoton,
            typename TOptionalSum, typename TOptionalMaxValues,
            typename TOptionalClusters, typename TOptionalUsedClusters>
  auto enqueueKernels(DeviceData<TConfig, TAlpaka> *dev, std::size_t numMaps,
                      ExecutionFlags flags, TData data, TOptionalEnergy energy,
                      TOptionalPhoton photon, TOptionalSum sum,
                      TOptionalMaxValues maxValues, TOptionalClusters clusters,
                      TOptionalUsedClusters usedClusters) -> void {
    typename TConfig::MaskMap *local_mask =
        flags.masking ? alpakaNativePtr(dev->mask) : nullptr;

    if (flags.mode == 0) {
      // converting to energy
      // the photon and cluster extraction kernels already include energy
      // conversion

      ConversionKernel<TConfig> conversionKernel{};
      auto const conversion(alpakaCreateKernel<TAlpaka>(
          getWorkDiv<TAlpaka>(), conversionKernel, alpakaNativePtr(data),
          alpakaNativePtr(dev->gain), alpakaNativePtr(dev->initialPedestal),
          alpakaNativePtr(dev->pedestal), alpakaNativePtr(dev->gainStage),
          alpakaNativePtr(energy), dev->numMaps, local_mask, pedestalFallback));

      DEBUG("enqueueing conversion kernel");
      alpakaEnqueueKernel(dev->queue, conversion);
    } else if (flags.mode == 1) {
      // converting to photons (and energy)
      PhotonFinderKernel<TConfig> photonFinderKernel{};
      auto const photonFinder(alpakaCreateKernel<TAlpaka>(
          getWorkDiv<TAlpaka>(), photonFinderKernel, alpakaNativePtr(data),
          alpakaNativePtr(dev->gain), alpakaNativePtr(dev->initialPedestal),
          alpakaNativePtr(dev->pedestal), alpakaNativePtr(dev->gainStage),
          alpakaNativePtr(energy), alpakaNativePtr(photon), dev->numMaps,
          beamConst, local_mask, pedestalFallback));

      DEBUG("enqueueing photon kernel");
      alpakaEnqueueKernel(dev->queue, photonFinder);
    } else {
      // clustering (and conversion to energy)

      // reset the number of clusters
      alpakaMemSet(dev->queue, usedClusters, 0,
                   decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

      for (uint32_t i = 0; i < numMaps + 1; ++i) {
        // execute the clusterfinder with the pedestal update on every
        // frame
        // execute cluster finder
        ClusterFinderKernel<TConfig, TAlpaka> clusterFinderKernel{};
        auto const clusterFinder(alpakaCreateKernel<TAlpaka>(
            getWorkDiv<TAlpaka>(), clusterFinderKernel, alpakaNativePtr(data),
            alpakaNativePtr(dev->gain), alpakaNativePtr(dev->initialPedestal),
            alpakaNativePtr(dev->pedestal), alpakaNativePtr(dev->gainStage),
            alpakaNativePtr(energy), alpakaNativePtr(clusters),
            alpakaNativePtr(usedClusters), local_mask, dev->numMaps, i,
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
            decltype(TAlpaka::threadsPerBlock)(TAlpaka::threadsPerBlock),
            static_cast<Size>(1));
        ReduceKernel<TAlpaka::threadsPerBlock, double> reduceKernelRun1;
        auto const reduceRun1(alpakaCreateKernel<TAlpaka>(
            workdivRun1, reduceKernelRun1, &alpakaNativePtr(energy)[i],
            &alpakaNativePtr(dev->maxValueMaps)[i],
            TConfig::DIMX * TConfig::DIMY));

        WorkDiv workdivRun2{
            static_cast<Size>(1),
            decltype(TAlpaka::threadsPerBlock)(TAlpaka::threadsPerBlock),
            static_cast<Size>(1)};
        ReduceKernel<TAlpaka::threadsPerBlock, double> reduceKernelRun2;
        auto const reduceRun2(alpakaCreateKernel<TAlpaka>(
            workdivRun2, reduceKernelRun2,
            &alpakaNativePtr(dev->maxValueMaps)[i],
            &alpakaNativePtr(dev->maxValueMaps)[i],
            decltype(TAlpaka::blocksPerGrid)(TAlpaka::blocksPerGrid)));
        alpakaEnqueueKernel(dev->queue, reduceRun1);
        alpakaEnqueueKernel(dev->queue, reduceRun2);
      }

      WorkDiv workdivMaxValueCopy{
          static_cast<Size>(std::ceil(
              (double)numMaps /
              decltype(TAlpaka::threadsPerBlock)(TAlpaka::threadsPerBlock))),
          decltype(TAlpaka::threadsPerBlock)(TAlpaka::threadsPerBlock),
          static_cast<Size>(1)};
      MaxValueCopyKernel<TConfig> maxValueCopyKernel{};
      auto const maxValueCopy(
          alpakaCreateKernel<TAlpaka>(workdivMaxValueCopy, maxValueCopyKernel,
                                      alpakaNativePtr(dev->maxValueMaps),
                                      alpakaNativePtr(maxValues), numMaps));

      DEBUG("enqueueing max value extraction kernel");
      alpakaEnqueueKernel(dev->queue, maxValueCopy);
    }

    // summation
    if (flags.summation) {
      DEBUG("enqueueing summation kernel");

      SummationKernel<TConfig> summationKernel{};
      auto const summation(alpakaCreateKernel<TAlpaka>(
          getWorkDiv<TAlpaka>(), summationKernel, alpakaNativePtr(energy),
          decltype(TConfig::SUM_FRAMES)(TConfig::SUM_FRAMES), dev->numMaps,
          alpakaNativePtr(sum)));

      alpakaEnqueueKernel(dev->queue, summation);
    };
  }
};
