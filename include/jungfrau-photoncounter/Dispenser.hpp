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

#define BOOST_OPTIONAL_USE_OLD_DEFINITION_OF_NONE
#include <boost/optional.hpp>

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
  Dispenser(typename TConfig::template FramePackage<typename TConfig::GainMap,
                                                    TAlpaka>
                gainMap,
            boost::optional<typename TAlpaka::template HostBuf<MaskMap>> mask)
      : gain(gainMap),
        mask((mask ? *mask
                   : alpakaAlloc<typename TConfig::MaskMap>(
                         alpakaGetHost<TAlpaka>(),
                         decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP)))),
        drift(alpakaAlloc<typename TConfig::DriftMap>(
            alpakaGetHost<TAlpaka>(),
            decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP))),
        gainStage(alpakaAlloc<typename TConfig::GainStageMap>(
            alpakaGetHost<TAlpaka>(),
            decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP))),
        maxValueMaps(alpakaAlloc<typename TConfig::EnergyMap>(
            alpakaGetHost<TAlpaka>(),
            decltype(TConfig::DEV_FRAMES)(TConfig::DEV_FRAMES))),
        pedestalFallback(false), init(false),
        ringbuffer(TAlpaka::STREAMS_PER_DEV * alpakaGetDevCount<TAlpaka>()),
        pedestal(TConfig::PEDEMAPS, alpakaGetHost<TAlpaka>()),
        initPedestal(TConfig::PEDEMAPS, alpakaGetHost<TAlpaka>()), nextFull(0),
        nextFree(0), deviceContainer(alpakaGetDevs<TAlpaka>()) {
    initDevices();

    // make room for live mask information
    if (!mask) {
      alpakaMemSet(devices[devices.size() - 1].queue,
                   devices[devices.size() - 1].mask, 1,
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
   * move assign constructor
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

  /**
   * Tries to upload all data packages requiered for the inital offset.
   * Only stops after all data packages are uploaded.
   * @param Maps-Struct with datamaps
   * @param stdDevThreshold An standard deviation threshold above which pixels
   * should be masked. If this is 0, no pixels will be masked.
   */
  auto uploadPedestaldata(typename TConfig::template FramePackage<
                              typename TConfig::DetectorData, TAlpaka>
                              data,
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
  }

  /**
   * Downloads the pedestal data.
   * @return pedestal pedestal data
   */
  auto downloadPedestaldata() ->
      typename TConfig::template FramePackage<typename TConfig::PedestalMap,
                                              TAlpaka> {
    // create handle for the device with the current version of the pedestal
    // maps
    auto current_device = devices[nextFree];

    DEBUG("downloading pedestaldata from device", nextFree);

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
  auto downloadInitialPedestaldata() ->
      typename TConfig::template FramePackage<typename TConfig::InitPedestalMap,
                                              TAlpaka> {
    // create handle for the device with the current version of the pedestal
    // maps
    auto current_device = devices[nextFree];

    DEBUG("downloading pedestaldata from device", nextFree);

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

    // create handle for the device with the current version of the pedestal
    // maps
    auto current_device = devices[nextFree];

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
  auto downloadGainStages(std::size_t frame = 0) ->
      typename TConfig::GainStageMap * {
    DEBUG("downloading gain stage map...");

    // create handle for the device with the current version of the pedestal
    // maps
    auto current_device = devices[nextFree];

    // mask gain stage maps
    GainStageMaskingKernel<TConfig> gainStageMasking;
    auto const gainStageMasker(alpakaCreateKernel<TAlpaka>(
        getWorkDiv<TAlpaka>(), gainStageMasking,
        alpakaNativePtr(current_device.gainStage),
        alpakaNativePtr(current_device.gainStageOutput), frame,
        alpakaNativePtr(current_device.mask)));

    alpakaEnqueueKernel(current_device.queue, gainStageMasker);

    // get the pedestal data from the device
    alpakaCopy(current_device.queue, gainStage, current_device.gainStageOutput,
               decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

    // wait for copy to finish
    alpakaWait(current_device.queue);

    return alpakaNativePtr(gainStage);
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

    // create handle for the device with the current version of the pedestal
    // maps
    auto current_device = devices[nextFree];

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

  //! @todo: remove this
  /**
   * Tries to upload one data package.
   * @param Maps-struct with raw data, offset within the package
   * @return number of frames uploaded from the package
   * /
  auto uploadData(
      typename TConfig::template FramePackage<typename TConfig::DetectorData,
                                              TAlpaka> data,
      std::size_t offset,
      typename TConfig::ExecutionFlags flags,
      bool flushWhenFinished = true)
      -> std::tuple<std::size_t, std::future<bool>>
  {
      if (!ringbuffer.isEmpty()) {
          // try uploading one data package
          if (offset <= data.numFrames - TConfig::DEV_FRAMES) {
              auto result = calcData(alpakaNativePtr(data.data) + offset,
                                     TConfig::DEV_FRAMES,
                                     flags);
              offset += std::get<0>(result);
              DEBUG(offset, "/", data.numFrames, "frames uploaded");

              return std::make_tuple(std::move(offset),
                                     std::move(std::get<1>(result)));
          }
          // upload remaining frames
          else if (offset != data.numFrames) {
              auto result = calcData(alpakaNativePtr(data.data) + offset,
                                     data.numFrames % TConfig::DEV_FRAMES,
                                     flags);
              DEBUG(offset, "/", data.numFrames, "frames uploaded");
              offset += std::get<0>(result);
              return std::make_tuple(std::move(offset),
                                     std::move(std::get<1>(result)));
          }
          // force wait for one device to finish since there's no new data and
          // the user wants the data flushed
          else if (flushWhenFinished) {
              DEBUG("flushing ...");

              flush();
          }
      }

      return std::make_tuple(
          offset, std::async(std::launch::async, []() { return true; }));
  }*/

  template <typename TFramePackageEnergyMap, typename TFramePackagePhotonMap,
            typename TFramePackageSumMap, typename TFramePackageEnergyValue>
  auto process(typename TConfig::template FramePackage<
                   typename TConfig::DetectorData, TAlpaka>
                   data,
               std::size_t offset, typename TConfig::ExecutionFlags flags,
               boost::optional<TFramePackageEnergyMap> energy,
               boost::optional<TFramePackagePhotonMap> photon,
               boost::optional<TFramePackageSumMap> sum,
               boost::optional<TFramePackageEnergyValue> maxValues,
               typename TConfig::template ClusterArray<TAlpaka> *clusters,
               bool flushWhenFinished = true)
      -> std::tuple<std::size_t, std::future<bool>> {
    //! @todo: fix this
    // if (!ringbuffer.isEmpty()) {
    // try uploading one data package
    if (offset <= data.numFrames - TConfig::DEV_FRAMES) {

      auto dataView = data.getView(offset, TConfig::DEV_FRAMES);

      auto result = processData(dataView, TConfig::DEV_FRAMES, flags, energy,
                                photon, sum, maxValues, clusters);
      offset += std::get<0>(result);
      DEBUG(offset, "/", data.numFrames, "frames uploaded");

      return std::make_tuple(std::move(offset), std::move(std::get<1>(result)));
    }
    // upload remaining frames
    else if (offset != data.numFrames) {

      auto dataView =
          data.getView(offset, data.numFrames % TConfig::DEV_FRAMES);
      auto result =
          processData(dataView, data.numFrames % TConfig::DEV_FRAMES, flags,
                      energy, photon, sum, maxValues, clusters);
      DEBUG(offset, "/", data.numFrames, "frames uploaded");
      offset += std::get<0>(result);
      return std::make_tuple(std::move(offset), std::move(std::get<1>(result)));
    }
    // force wait for one device to finish since there's no new data and
    // the user wants the data flushed
    else if (flushWhenFinished) {
      DEBUG("flushing ...");

      flush();
    }
    //}

    return std::make_tuple(
        offset, std::async(std::launch::async, []() { return true; }));
  }

  //! @todo: remove this
  /**
   * Tries to download one data package.
   * @param pointer to empty struct for photon and sum maps and cluster data
   * @return boolean indicating whether maps were downloaded or not
   * /
  template <typename TFramePackageEnergyMap,
            typename TFramePackagePhotonMap,
            typename TFramePackageSumMap,
            typename TFramePackageEnergyValue>
  auto
  downloadData(boost::optional<TFramePackageEnergyMap> energy,
               boost::optional<TFramePackagePhotonMap> photon,
               boost::optional<TFramePackageSumMap> sum,
               boost::optional<TFramePackageEnergyValue> maxValues,
               typename TConfig::template ClusterArray<TAlpaka>* clusters)
      -> std::tuple<size_t, std::future<bool>>
  {
      // get the oldest finished device
      struct DeviceData<TConfig, TAlpaka>* dev =
          &Dispenser::devices[nextFree];

      // to keep frames in order only download if the longest running device
      // has finished
      if (dev->state != READY)
          return std::make_tuple(
              0, std::async(std::launch::async, []() { return true; }));

      // download energy if needed
      if (energy) {
          DEBUG("downloading energy");
          (*energy).numFrames = dev->numMaps;
          alpakaCopyIfNeeded(dev->queue, (*energy).data, dev->energy,
  dev->numMaps);
      }

      // download photon data if needed
      if (photon) {
          DEBUG("downloading photons");
          (*photon).numFrames = dev->numMaps;
          alpakaCopyIfNeeded(dev->queue, (*photon).data, dev->photon,
  dev->numMaps);
      }

      // download summation frames if needed
      if (sum) {
          DEBUG("downloading sum");
          (*sum).numFrames = dev->numMaps / TConfig::SUM_FRAMES;
          alpakaCopyIfNeeded(dev->queue,
                     (*sum).data,
                     dev->sum,
                     (dev->numMaps / TConfig::SUM_FRAMES));
      }

      // download maximum values if needed
      if (maxValues) {
          DEBUG("downloading max values");
          (*maxValues).numFrames = dev->numMaps;
          alpakaCopyIfNeeded(dev->queue,
                     (*maxValues).data,
                     dev->maxValues,
                     decltype(TConfig::DEV_FRAMES)(TConfig::DEV_FRAMES));
      }

      // download number of clusters if needed
      if (clusters) {
          // download number of found clusters
          DEBUG("downloading clusters");
          alpakaCopy(dev->queue,
                     (*clusters).usedPinned,
                     dev->numClusters,
                     decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

          // wait for completion of copy operations
          alpakaWait(dev->queue);

          // reserve space in the cluster array for the new data and save
          // the number of clusters to download temporarily
          auto oldNumClusters = (*clusters).used;


          DEBUG("total current clusters:", oldNumClusters);


          auto clustersToDownload =
              alpakaNativePtr((*clusters).usedPinned)[0];
          alpakaNativePtr((*clusters).usedPinned)[0] += oldNumClusters;
          (*clusters).used = alpakaNativePtr((*clusters).usedPinned)[0];

          DEBUG("Downloading ", clustersToDownload, "clusters. ");
          DEBUG("Total downloaded clusters:", (*clusters).used);

          // create a subview in the cluster buffer where the new data
          // shuld be downloaded to
          auto const extentView(Vec(static_cast<Size>(clustersToDownload)));
          auto const offsetView(Vec(static_cast<Size>(oldNumClusters)));
          typename TAlpaka::template HostView<typename TConfig::Cluster>
              clusterView(clusters->clusters, extentView, offsetView);

          // download actual clusters
          alpakaCopy(
              dev->queue, clusterView, dev->cluster, clustersToDownload);
      }

      // free the device
      dev->state = FREE;
      nextFree = (nextFree + 1) % devices.size();
      ringbuffer.push(dev);

      // create a future, that waits for the copying to be finished
      auto wait = [](decltype(dev) dev) {
          alpakaWait(dev->queue);
          return true;
      };

      // return the number of downloaded frames and the future
      return std::make_tuple(dev->numMaps,
                             std::async(std::launch::async, wait, dev));
  }
  */

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
  typename TConfig::template FramePackage<typename TConfig::GainMap, TAlpaka>
      gain;
  typename TAlpaka::template HostBuf<typename TConfig::MaskMap> mask;
  typename TAlpaka::template HostBuf<typename TConfig::DriftMap> drift;
  typename TAlpaka::template HostBuf<typename TConfig::GainStageMap> gainStage;
  typename TAlpaka::template HostBuf<typename TConfig::EnergyMap> maxValueMaps;

  typename TConfig::template FramePackage<typename TConfig::PedestalMap,
                                          TAlpaka>
      pedestal;
  typename TConfig::template FramePackage<typename TConfig::InitPedestalMap,
                                          TAlpaka>
      initPedestal;

  std::vector<typename TAlpaka::DevAcc> deviceContainer;

  bool init;
  bool pedestalFallback;
  Ringbuffer<DeviceData<TConfig, TAlpaka> *> ringbuffer;
  std::vector<DeviceData<TConfig, TAlpaka>> devices;

  std::size_t nextFree, nextFull;

  /**
   * Initializes all devices. Uploads gain data and creates buffer.
   * @param vector with devices to be initialized
   */
  auto initDevices() -> void {
    const GainmapInversionKernel<TConfig> gainmapInversionKernel{};
    std::size_t deviceCount = alpakaGetDevCount<TAlpaka>();
    devices.reserve(deviceCount * TAlpaka::STREAMS_PER_DEV);

    for (std::size_t num = 0; num < deviceCount * TAlpaka::STREAMS_PER_DEV;
         ++num) {
      // initialize variables
      devices.emplace_back(num,
                           &deviceContainer[num / TAlpaka::STREAMS_PER_DEV]);
      alpakaCopy(devices[num].queue, devices[num].gain, gain.data,
                 decltype(TConfig::GAINMAPS)(TConfig::GAINMAPS));

      // compute reciprocals of gain maps
      auto const gainmapInversion(alpakaCreateKernel<TAlpaka>(
          getWorkDiv<TAlpaka>(), gainmapInversionKernel,
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

    // copy offset data from last initialized device
    auto prevDevice = (nextFull + devices.size() - 1) % devices.size();
    alpakaWait(devices[prevDevice].queue);
    alpakaCopy(dev->queue, dev->pedestal, devices[prevDevice].pedestal,
               decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

    alpakaCopy(dev->queue, dev->initialPedestal,
               devices[prevDevice].initialPedestal,
               decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

    alpakaCopy(dev->queue, dev->mask, devices[prevDevice].mask,
               decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

    // increase nextFull and nextFree (because pedestal data isn't
    // downloaded like normal data)
    nextFull = (nextFull + 1) % devices.size();
    nextFree = (nextFree + 1) % devices.size();

    if (!init) {
      alpakaMemSet(dev->queue, dev->pedestal, 0,
                   decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));
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
                 decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

      // distribute pedestal map (with initial data)
      alpakaCopy(devices[source].queue, devices[i].pedestal,
                 devices[source].pedestal,
                 decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));
    }
    synchronize();
  }

  //! @todo: remove this
  /**
   * Executes summation and correction kernel.
   * @param pointer to raw data and number of frames
   * @return number of frames calculated
   * /
template <typename TDetectorData,
          typename TFramePackageEnergyMap,
            typename TFramePackagePhotonMap,
            typename TFramePackageSumMap,
            typename TFramePackageEnergyValue>
  auto calcData(TDetectorData* data,
                std::size_t numMaps,
                typename TConfig::ExecutionFlags flags,
                boost::optional<TFramePackageEnergyMap> energy,
               boost::optional<TFramePackagePhotonMap> photon,
               boost::optional<TFramePackageSumMap> sum,
               boost::optional<TFramePackageEnergyValue> maxValues,
               typename TConfig::template ClusterArray<TAlpaka>* clusters)
      -> std::tuple<std::size_t, std::future<bool>>
  {
      // get the next free device from the ringbuffer
      DeviceData<TConfig, TAlpaka>* dev;
      if (!ringbuffer.pop(dev))
          return std::make_tuple(
              0, std::async(std::launch::async, []() { return true; }));

      // set the state to processing
      dev->state = PROCESSING;
      dev->numMaps = numMaps;

      // upload the data to the device
      alpakaCopyIfNeeded(
          dev->queue,
          dev->data,
          alpakaViewPlainPtrHost<TAlpaka, typename TConfig::DetectorData>(
              data, alpakaGetHost<TAlpaka>(), numMaps),
          numMaps);

      // copy offset data from last device uploaded to
      auto prevDevice = (nextFull + devices.size() - 1) % devices.size();
      alpakaWait(dev->queue, devices[prevDevice].event);
      DEBUG("device", devices[prevDevice].id, "finished");

      devices[prevDevice].state = READY;
      alpakaCopy(dev->queue,
                 dev->pedestal,
                 devices[prevDevice].pedestal,
                 decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

      nextFull = (nextFull + 1) % devices.size();

      typename TConfig::MaskMap* local_mask =
          flags.masking ? alpakaNativePtr(dev->mask) : nullptr;

      if (flags.mode == 0) {
          // converting to energy
          // the photon and cluster extraction kernels already include energy
          // conversion
          ConversionKernel<TConfig> conversionKernel{};
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
      else if (flags.mode == 1) {
          // converting to photons (and energy)
          PhotonFinderKernel<TConfig> photonFinderKernel{};
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
      }
      else {
          // clustering (and conversion to energy)
          DEBUG("enqueueing clustering kernel");

          // reset the number of clusters
          alpakaMemSet(dev->queue,
                       dev->numClusters,
                       0,
                       decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

          for (uint32_t i = 0; i < numMaps + 1; ++i) {
              // execute the clusterfinder with the pedestalupdate on every
              // frame
              ClusterFinderKernel<TConfig> clusterFinderKernel{};
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
                  TConfig::DIMX * TConfig::DIMY));

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
          MaxValueCopyKernel<TConfig> maxValueCopyKernel{};
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

          SummationKernel<TConfig> summationKernel{};
          auto const summation(alpakaCreateKernel<TAlpaka>(
              getWorkDiv<TAlpaka>(),
              summationKernel,
              alpakaNativePtr(dev->energy),
              decltype(TConfig::SUM_FRAMES)(TConfig::SUM_FRAMES),
              dev->numMaps,
              alpakaNativePtr(dev->sum)));

          alpakaEnqueueKernel(dev->queue, summation);
      };

      // the event is used to wait for pedestal data
      alpakaEnqueueKernel(dev->queue, dev->event);






      // download energy if needed
      if (energy) {
          DEBUG("downloading energy");
          (*energy).numFrames = dev->numMaps;
          alpakaCopyIfNeeded(dev->queue, (*energy).data, dev->energy,
dev->numMaps);
      }

      // download photon data if needed
      if (photon) {
          DEBUG("downloading photons");
          (*photon).numFrames = dev->numMaps;
          alpakaCopyIfNeeded(dev->queue, (*photon).data, dev->photon,
dev->numMaps);
      }

      // download summation frames if needed
      if (sum) {
          DEBUG("downloading sum");
          (*sum).numFrames = dev->numMaps / TConfig::SUM_FRAMES;
          alpakaCopyIfNeeded(dev->queue,
                     (*sum).data,
                     dev->sum,
                     (dev->numMaps / TConfig::SUM_FRAMES));
      }

      // download maximum values if needed
      if (maxValues) {
          DEBUG("downloading max values");
          (*maxValues).numFrames = dev->numMaps;
          alpakaCopyIfNeeded(dev->queue,
                     (*maxValues).data,
                     dev->maxValues,
                     decltype(TConfig::DEV_FRAMES)(TConfig::DEV_FRAMES));
      }

      // download number of clusters if needed
      if (clusters) {
          // download number of found clusters
          DEBUG("downloading clusters");
          alpakaCopy(dev->queue,
                     (*clusters).usedPinned,
                     dev->numClusters,
                     decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

          // wait for completion of copy operations
          alpakaWait(dev->queue);

          // reserve space in the cluster array for the new data and save
          // the number of clusters to download temporarily
          auto oldNumClusters = (*clusters).used;


          DEBUG("total current clusters:", oldNumClusters);


          auto clustersToDownload =
              alpakaNativePtr((*clusters).usedPinned)[0];
          alpakaNativePtr((*clusters).usedPinned)[0] += oldNumClusters;
          (*clusters).used = alpakaNativePtr((*clusters).usedPinned)[0];

          DEBUG("Downloading ", clustersToDownload, "clusters. ");
          DEBUG("Total downloaded clusters:", (*clusters).used);

          // create a subview in the cluster buffer where the new data
          // shuld be downloaded to
          auto const extentView(Vec(static_cast<Size>(clustersToDownload)));
          auto const offsetView(Vec(static_cast<Size>(oldNumClusters)));
          typename TAlpaka::template HostView<typename TConfig::Cluster>
              clusterView(clusters->clusters, extentView, offsetView);

          // download actual clusters
          alpakaCopy(
              dev->queue, clusterView, dev->cluster, clustersToDownload);
      }

      // free the device
      dev->state = FREE;
      nextFree = (nextFree + 1) % devices.size();
      ringbuffer.push(dev);









      auto wait = [](decltype(dev) dev) {
          alpakaWait(dev->queue);
          return true;
      };

      return std::make_tuple(numMaps,
                             std::async(std::launch::async, wait, dev));


}
*/

#include <type_traits>

  //! @todo: use ptr like stuff instead of FramePackages here!
  template <typename TFramePackageDetectorData, typename TFramePackageEnergyMap,
            typename TFramePackagePhotonMap, typename TFramePackageSumMap,
            typename TFramePackageEnergyValue>
  auto processData(TFramePackageDetectorData data, std::size_t numMaps,
                   typename TConfig::ExecutionFlags flags,
                   boost::optional<TFramePackageEnergyMap> energy,
                   boost::optional<TFramePackagePhotonMap> photon,
                   boost::optional<TFramePackageSumMap> sum,
                   boost::optional<TFramePackageEnergyValue> maxValues,
                   typename TConfig::template ClusterArray<TAlpaka> *clusters)
      -> std::tuple<std::size_t, std::future<bool>> {

    using ClusterView =
        typename TAlpaka::template HostView<typename TConfig::Cluster>;

    DeviceData<TConfig, TAlpaka> *dev = &devices[nextFree];
    dev->numMaps = numMaps;

    // create a shadow buffer on the device and upload data if required
    typename TConfig::template FramePackageDoubleBuffer<
        TAlpaka, TFramePackageDetectorData, decltype(dev->data)>
        dataBuffer(data, dev->data, &dev->queue, numMaps);
    dataBuffer.upload();

    // create shadow buffers on the device if required
    typename TConfig::template FramePackageDoubleBuffer<
        TAlpaka, TFramePackageEnergyMap, decltype(dev->energy)>
        energyBuffer(energy, dev->energy, &dev->queue, numMaps);
    typename TConfig::template FramePackageDoubleBuffer<
        TAlpaka, TFramePackagePhotonMap, decltype(dev->photon)>
        photonBuffer(photon, dev->photon, &dev->queue, numMaps);
    typename TConfig::template FramePackageDoubleBuffer<
        TAlpaka, TFramePackageSumMap, decltype(dev->sum)>
        sumBuffer(sum, dev->sum, &dev->queue, numMaps / TConfig::SUM_FRAMES);
    typename TConfig::template FramePackageDoubleBuffer<
        TAlpaka, TFramePackageEnergyValue, decltype(dev->maxValues)>
        maxValuesBuffer(maxValues, dev->maxValues, &dev->queue, numMaps);

    // convert pointer to boost::optional
    boost::optional<ClusterView> optionalClusters;
    boost::optional<decltype(clusters->usedPinned)> optionalNumClusters;
    if (clusters) {
      // create subview of the free part of the cluster array

      ClusterView c(clusters->clusters, clusters->used,
                    alpakaGetExtent<0>(clusters->clusters) - clusters->used);

      /*optionalClusters.emplace(
          ClusterView(clusters->clusters, clusters->used,
                      alpakaGetExtent<0>(clusters->clusters) -
         clusters->used));*/
      optionalClusters.emplace(c);
      optionalNumClusters = clusters->usedPinned;
    }

    // temporarily set clusterBuffer size to 0 because the correct number of
    // clusters to download is not yet known
    typename TConfig::template GetDoubleBuffer<TAlpaka, ClusterView,
                                               decltype(dev->cluster)>::Buffer
        clusterBuffer(optionalClusters, dev->cluster, &dev->queue,
                      static_cast<std::size_t>(0));
    typename TConfig::template GetDoubleBuffer<
        TAlpaka, decltype(clusters->usedPinned),
        decltype(dev->numClusters)>::Buffer
        clustersUsedBuffer(optionalNumClusters, dev->numClusters, &dev->queue,
                           TConfig::SINGLEMAP);

    //! @todo: remove this
    // uploadData(dev, data,
    // energy, photon, sum,
    // maxValues, clusters,
    // numMaps);

    // copy offset data from last device uploaded to current device
    auto prevDevice = (nextFull + devices.size() - 1) % devices.size();
    alpakaWait(dev->queue, devices[prevDevice].event);
    DEBUG("device", devices[prevDevice].id, "finished");

    devices[prevDevice].state = READY;
    alpakaCopy(dev->queue, dev->pedestal, devices[prevDevice].pedestal,
               decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

    nextFull = (nextFull + 1) % devices.size();

    // enqueue the kernels
    enqueueKernels(dev, numMaps, flags, dataBuffer, energyBuffer, photonBuffer,
                   sumBuffer, maxValuesBuffer, clusterBuffer,
                   clustersUsedBuffer);

    // the event is used to wait
    // for pedestal data
    alpakaEnqueueKernel(dev->queue, dev->event);
    //! @todo: remove this
    // enqueue download of the
    // data downloadData(dev,
    // energy, photon, sum,
    // maxValues, clusters);

    // download the data
    energyBuffer.download();
    photonBuffer.download();
    sumBuffer.download();
    maxValuesBuffer.download();
    if (clusters) {
      //! @todo: download correct amount of clusters
      clustersUsedBuffer.download();

      // wait for completion of copy operations
      alpakaWait(dev->queue);

      auto oldNumClusters = (*clusters).used;
      DEBUG("total current clusters:", oldNumClusters);

      auto clustersToDownload = alpakaNativePtr((*clusters).usedPinned)[0];
      alpakaNativePtr((*clusters).usedPinned)[0] += oldNumClusters;
      (*clusters).used = alpakaNativePtr((*clusters).usedPinned)[0];

      DEBUG("Downloading ", clustersToDownload, "clusters. ");
      DEBUG("Total downloaded clusters:", (*clusters).used);

      //! @todo: find a better solution for this
      clusterBuffer.setSize(alpakaNativePtr(clusters->usedPinned)[0]);

      clusterBuffer.download();
    }

    // update the nextFree index
    nextFree = (nextFree + 1) % devices.size();

    auto wait = [](decltype(dev) dev) {
      alpakaWait(dev->queue);
      return true;
    };

    return std::make_tuple(numMaps, std::async(std::launch::async, wait, dev));
  }

  //! @todo: remove this
  /*
  template <typename TDetectorData,
            typename TFramePackageEnergyMap,
              typename TFramePackagePhotonMap,
              typename TFramePackageSumMap,
              typename TFramePackageEnergyValue>
    auto
    uploadData(DeviceData<TConfig, TAlpaka>* dev,
               TDetectorData* data,
                 boost::optional<TFramePackageEnergyMap> energy,
                 boost::optional<TFramePackagePhotonMap> photon,
                 boost::optional<TFramePackageSumMap> sum,
                 boost::optional<TFramePackageEnergyValue> maxValues,
               typename TConfig::template ClusterArray<TAlpaka>* clusters,
               std::size_t numMaps)
  {
    // upload the data to the device
        return alpakaCopyIfNeeded<TAlpaka>(
            dev->queue,
            dev->data,
            alpakaViewPlainPtrHost<TAlpaka, typename TConfig::DetectorData>(
                data, alpakaGetHost<TAlpaka>(), numMaps),
            numMaps);


        //! @todo: problems occur if data is only a buffer view
        //! @todo: implement extra view for processing




        // synchronize host and device buffers
        /*if(energy)
          alpakaSyncHostBuffers<TAlpaka>(dev->energy, energy);
        if(photon)
          alpakaSyncHostBuffers<TAlpaka>(dev->photon, photon);
        if(sum)
          alpakaSyncHostBuffers<TAlpaka>(dev->sum, sum);
        if(maxValues)
          alpakaSyncHostBuffers<TAlpaka>(dev->maxValues, maxValues);
        if(clusters) {
          alpakaSyncHostBuffers<TAlpaka>(dev->cluster, clusters->clusters);
          alpakaSyncHostBuffers<TAlpaka>(dev->numClusters,
  clusters->usedPinned);
          }* /
          }*/

  template <typename TData, typename TOptionalEnergy, typename TOptionalPhoton,
            typename TOptionalSum, typename TOptionalMaxValues,
            typename TOptionalClusters, typename TOptionalUsedClusters>
  auto enqueueKernels(DeviceData<TConfig, TAlpaka> *dev, std::size_t numMaps,
                      typename TConfig::ExecutionFlags flags, TData data,
                      TOptionalEnergy energy, TOptionalPhoton photon,
                      TOptionalSum sum, TOptionalMaxValues maxValues,
                      TOptionalClusters clusters,
                      TOptionalUsedClusters usedClusters) -> void {
    typename TConfig::MaskMap *local_mask =
        flags.masking ? alpakaNativePtr(dev->mask) : nullptr;

    // generate local variables (required for alpakaNativePtr)
    auto dataPtr = data.get();
    auto energyPtr = energy.get();
    auto photonPtr = photon.get();
    auto sumPtr = sum.get();
    auto maxValuesPtr = maxValues.get();
    auto clustersPtr = clusters.get();
    auto usedClustersPtr = usedClusters.get();

    if (flags.mode == 0) {
      // converting to energy
      // the photon and cluster extraction kernels already include energy
      // conversion

      ConversionKernel<TConfig> conversionKernel{};
      auto const conversion(alpakaCreateKernel<TAlpaka>(
          getWorkDiv<TAlpaka>(), conversionKernel, alpakaNativePtr(dataPtr),
          alpakaNativePtr(dev->gain), alpakaNativePtr(dev->initialPedestal),
          alpakaNativePtr(dev->pedestal), alpakaNativePtr(dev->gainStage),
          alpakaNativePtr(energyPtr), dev->numMaps, local_mask,
          pedestalFallback));

      DEBUG("enqueueing conversion kernel");
      alpakaEnqueueKernel(dev->queue, conversion);
    } else if (flags.mode == 1) {
      // converting to photons (and energy)
      PhotonFinderKernel<TConfig> photonFinderKernel{};
      auto const photonFinder(alpakaCreateKernel<TAlpaka>(
          getWorkDiv<TAlpaka>(), photonFinderKernel, alpakaNativePtr(dataPtr),
          alpakaNativePtr(dev->gain), alpakaNativePtr(dev->initialPedestal),
          alpakaNativePtr(dev->pedestal), alpakaNativePtr(dev->gainStage),
          alpakaNativePtr(energyPtr), alpakaNativePtr(photonPtr), dev->numMaps,
          local_mask, pedestalFallback));

      DEBUG("enqueueing photon kernel");
      alpakaEnqueueKernel(dev->queue, photonFinder);
    } else {
      // clustering (and conversion to energy)
      DEBUG("enqueueing clustering kernel");

      // reset the number of clusters
      alpakaMemSet(dev->queue, usedClustersPtr, 0,
                   decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

      for (uint32_t i = 0; i < numMaps + 1; ++i) {
        // execute the clusterfinder with the pedestalupdate on every
        // frame
        ClusterFinderKernel<TConfig> clusterFinderKernel{};
        auto const clusterFinder(alpakaCreateKernel<TAlpaka>(
            getWorkDiv<TAlpaka>(), clusterFinderKernel,
            alpakaNativePtr(dataPtr), alpakaNativePtr(dev->gain),
            alpakaNativePtr(dev->initialPedestal),
            alpakaNativePtr(dev->pedestal), alpakaNativePtr(dev->gainStage),
            alpakaNativePtr(energyPtr), alpakaNativePtr(clustersPtr),
            alpakaNativePtr(usedClustersPtr), local_mask, dev->numMaps, i,
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
            workdivRun1, reduceKernelRun1, &alpakaNativePtr(energyPtr)[i],
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
                                      alpakaNativePtr(maxValuesPtr), numMaps));

      DEBUG("enqueueing max value extraction kernel");
      alpakaEnqueueKernel(dev->queue, maxValueCopy);
    }

    // summation
    if (flags.summation) {
      DEBUG("enqueueing summation kernel");

      SummationKernel<TConfig> summationKernel{};
      auto const summation(alpakaCreateKernel<TAlpaka>(
          getWorkDiv<TAlpaka>(), summationKernel, alpakaNativePtr(energyPtr),
          decltype(TConfig::SUM_FRAMES)(TConfig::SUM_FRAMES), dev->numMaps,
          alpakaNativePtr(sumPtr)));

      alpakaEnqueueKernel(dev->queue, summation);
    };
  }

  //! @todo: remove this
  /*    template <typename TFramePackageEnergyMap,
              typename TFramePackagePhotonMap,
              typename TFramePackageSumMap,
              typename TFramePackageEnergyValue>
    auto
    downloadData(DeviceData<TConfig, TAlpaka>* dev,
                 boost::optional<TFramePackageEnergyMap> energy,
                 boost::optional<TFramePackagePhotonMap> photon,
                 boost::optional<TFramePackageSumMap> sum,
                 boost::optional<TFramePackageEnergyValue> maxValues,
                 typename TConfig::template ClusterArray<TAlpaka>* clusters)
        -> void
    {
        // download energy if needed
        if (energy) {
            DEBUG("downloading energy");
            (*energy).numFrames = dev->numMaps;
            alpakaCopyIfNeeded<TAlpaka>(dev->queue, (*energy).data, dev->energy,
    dev->numMaps);
        }

        // download photon data if needed
        if (photon) {
            DEBUG("downloading photons");
            (*photon).numFrames = dev->numMaps;
            alpakaCopyIfNeeded<TAlpaka>(dev->queue, (*photon).data, dev->photon,
    dev->numMaps);
        }

        // download summation frames if needed
        if (sum) {
            DEBUG("downloading sum");
            (*sum).numFrames = dev->numMaps / TConfig::SUM_FRAMES;
            alpakaCopyIfNeeded<TAlpaka>(dev->queue,
                       (*sum).data,
                       dev->sum,
                       (dev->numMaps / TConfig::SUM_FRAMES));
        }

        // download maximum values if needed
        if (maxValues) {
            DEBUG("downloading max values");
            (*maxValues).numFrames = dev->numMaps;
            alpakaCopyIfNeeded<TAlpaka>(dev->queue,
                       (*maxValues).data,
                       dev->maxValues,
                       decltype(TConfig::DEV_FRAMES)(TConfig::DEV_FRAMES));
        }

        // download number of clusters if needed
        if (clusters) {
            // download number of found clusters
            DEBUG("downloading clusters");
            alpakaCopy(dev->queue,
                       (*clusters).usedPinned,
                       dev->numClusters,
                       decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

            // wait for completion of copy operations
            alpakaWait(dev->queue);

            // reserve space in the cluster array for the new data and save
            // the number of clusters to download temporarily
            auto oldNumClusters = (*clusters).used;


            DEBUG("total current clusters:", oldNumClusters);


            auto clustersToDownload =
                alpakaNativePtr((*clusters).usedPinned)[0];
            alpakaNativePtr((*clusters).usedPinned)[0] += oldNumClusters;
            (*clusters).used = alpakaNativePtr((*clusters).usedPinned)[0];

            DEBUG("Downloading ", clustersToDownload, "clusters. ");
            DEBUG("Total downloaded clusters:", (*clusters).used);

            // create a subview in the cluster buffer where the new data
            // shuld be downloaded to
            auto const extentView(Vec(static_cast<Size>(clustersToDownload)));
            auto const offsetView(Vec(static_cast<Size>(oldNumClusters)));
            typename TAlpaka::template HostView<typename TConfig::Cluster>
                clusterView(clusters->clusters, extentView, offsetView);

            // download actual clusters
            alpakaCopy(
                dev->queue, clusterView, dev->cluster, clustersToDownload);
        }
        }*/
};
