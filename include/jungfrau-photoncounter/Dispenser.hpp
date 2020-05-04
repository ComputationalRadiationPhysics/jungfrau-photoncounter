#pragma once

#include "Alpakaconfig.hpp"
#include "Config.hpp"
#include "Ringbuffer.hpp"
#include "deviceData.hpp"

#include "kernel/Calibration.hpp"
#include "kernel/ClusterFinder.hpp"
#include "kernel/GainStageMasking.hpp"
#include "kernel/GainmapInversion.hpp"

#include <optional.hpp>

#include <future>
#include <iostream>


template<typename T>
void write_img(std::string path, T* data, std::size_t header, std::size_t elemSize = sizeof(T)) {
  std::ofstream img(path + ".dat", std::ios::binary);
  std::size_t size = 1024 * 512 * elemSize + header;
  
  img.write(reinterpret_cast<const char*>(data), size);
    
  img.flush();
  img.close();
}


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
			  unsigned int moduleNumber = 0,
			  unsigned int moduleCount = 1)
        : gain(gainMap),
          mask((mask ? *mask
                     : alpakaAlloc<typename TConfig::MaskMap>(
                           alpakaGetHost<TAlpaka>(),
                           decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP)))),
          gainStage(decltype(TConfig::DEV_FRAMES)(TConfig::DEV_FRAMES)),
          init(false),
          pedestal(TConfig::PEDEMAPS, alpakaGetHost<TAlpaka>()),
          initPedestal(TConfig::PEDEMAPS, alpakaGetHost<TAlpaka>()),
          beamConst(beamConst),
          deviceContainer(alpakaGetDevs<TAlpaka>()),
          device(0, &deviceContainer[0])
    {
        initDevices();

        if (!mask) {
          DEBUG("Creating new mask map on device", 0);
          alpakaMemSet(device.queue,
                       device.mask,
                       1,
                       decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));
        } else {
          DEBUG("Loading existing mask map on device", 0);
          alpakaCopy(device.queue,
                     device.mask,
                     *mask,
                     decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));
        }

        alpakaWait(device.queue);
    }

    auto synchronize() -> void
    {
        alpakaWait(device.queue);
    }

    auto reset() -> void
    {
        // reset variables
        init = false;
        
        // init devices
        initDevices();

        // clear mask
        alpakaMemSet(device.queue,
                     device.mask,
                     1,
                     decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

        // synchronize
        alpakaWait(device.queue);
    }

  auto uploadPedestaldata(
        FramePackage<typename TConfig::DetectorData, TAlpaka> data,
        double stdDevThreshold = 0) -> void
    {
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
    }

  auto uploadRawPedestals(FramePackage<typename TConfig::InitPedestalMap, TAlpaka> data) -> void
    {
        DEBUG("distribute raw pedestal maps");
        alpakaCopy(device.queue,
                   device.initialPedestal,
                   data.data,
                   decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));
        alpakaWait(device.queue);
    }

  auto downloadPedestaldata()
        -> FramePackage<typename TConfig::PedestalMap, TAlpaka>
    {
        DEBUG("downloading pedestaldata from device", 0);

        // get the pedestal data from the device
        alpakaCopy(device.queue,
                   pedestal.data,
                   device.pedestal,
                   decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

        // wait for copy to finish
        alpakaWait(device.queue);

        pedestal.numFrames = TConfig::PEDEMAPS;

        return pedestal;
    }

    auto downloadInitialPedestaldata()
        -> FramePackage<typename TConfig::InitPedestalMap, TAlpaka>
    {   
        DEBUG("downloading pedestaldata from device", 0);

        // get the pedestal data from the device
        alpakaCopy(device.queue,
                   initPedestal.data,
                   device.initialPedestal,
                   decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

        // wait for copy to finish
        alpakaWait(device.queue);

        initPedestal.numFrames = TConfig::PEDEMAPS;

        return initPedestal;
    }

    auto downloadMask() -> typename TConfig::MaskMap*
    {
        DEBUG("downloading mask...");

        // get the pedestal data from the device
        alpakaCopy(device.queue,
                   mask,
                   device.mask,
                   decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

        // wait for copy to finish
        alpakaWait(device.queue);

        return alpakaNativePtr(mask);
    }

    auto downloadGainStages()
        -> FramePackage<typename TConfig::GainStageMap, TAlpaka>
    {
        DEBUG("downloading gain stage map...");

        // mask gain stage maps
        GainStageMaskingKernel<TConfig> gainStageMasking;
        auto const gainStageMasker(alpakaCreateKernel<TAlpaka>(
            getWorkDiv<TAlpaka>(),
            gainStageMasking,
            alpakaNativePtr(device.gainStage),
            alpakaNativePtr(device.gainStageOutput),
            device.numMaps,
            alpakaNativePtr(device.mask)));

        alpakaEnqueueKernel(device.queue, gainStageMasker);

        // get the pedestal data from the device
        alpakaCopy(device.queue,
                   gainStage.data,
                   device.gainStageOutput,
                   device.numMaps);

        // wait for copy to finish
        alpakaWait(device.queue);

        gainStage.numFrames = device.numMaps;

        return gainStage;
    }

    template <typename TFramePackageEnergyMap,
              typename TFramePackagePhotonMap>
    auto process(FramePackage<typename TConfig::DetectorData, TAlpaka> data,
                 std::size_t offset,
                 tl::optional<TFramePackageEnergyMap> energy,
                 tl::optional<TFramePackagePhotonMap> photon,
                 typename TConfig::template ClusterArray<TAlpaka>* clusters)
        -> std::tuple<std::size_t, std::future<bool>>
    {

      auto numMaps = data.numFrames % (TConfig::DEV_FRAMES + 1);

        using ClusterView =
            typename TAlpaka::template HostView<typename TConfig::Cluster>;

        //! @todo: pass numMaps through data.numFrames??
        device.numMaps = numMaps;

        // create a shadow buffer on the device and upload data if required
        FramePackageDoubleBuffer<TAlpaka,
                                 //TFramePackageDetectorData,
                                 FramePackage<typename TConfig::DetectorData, TAlpaka>,
                                 decltype(device.data)>
            dataBuffer(data, device.data, &device.queue, numMaps);
        dataBuffer.upload();

        // create shadow buffers on the device if required
        FramePackageDoubleBuffer<TAlpaka,
                                 TFramePackageEnergyMap,
                                 decltype(device.energy)>
            energyBuffer(energy, device.energy, &device.queue, numMaps);

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
        typename GetDoubleBuffer<TAlpaka, ClusterView, decltype(device.cluster)>::
            Buffer clusterBuffer(optionalClusters,
                                 device.cluster,
                                 &device.queue,
                                 static_cast<std::size_t>(0));
        typename GetDoubleBuffer<TAlpaka,
                                 decltype(clusters->usedPinned),
                                 decltype(device.numClusters)>::Buffer
            clustersUsedBuffer(optionalNumClusters,
                               device.numClusters,
                               &device.queue,
                               TConfig::SINGLEMAP);

        typename TConfig::MaskMap* local_mask = alpakaNativePtr(device.mask);

        // clustering (and conversion to energy)
        DEBUG("enqueueing clustering kernel");

        auto usedClusters = clustersUsedBuffer.get();
        auto clustersBufferResult = clusterBuffer.get();
        auto energyBufferResult = energyBuffer.get();
        
        // reset the number of clusters
        alpakaMemSet(device.queue,
                     //usedClusters,
                     usedClusters,
                     0,
                     decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

        for (uint32_t i = 0; i < numMaps; ++i) {
          // execute the clusterfinder with the pedestal update on every frame
          // execute energy conversion
              
          ClusterEnergyKernel<TConfig, TAlpaka> clusterEnergyKernel{};
          auto const clusterEnergy(alpakaCreateKernel<TAlpaka>(getWorkDiv<TAlpaka>(),
                                                               clusterEnergyKernel,
                                                               alpakaNativePtr(dataBuffer.get()),
                                                               alpakaNativePtr(device.gain),
                                                               alpakaNativePtr(device.initialPedestal),
                                                               alpakaNativePtr(device.pedestal),
                                                               alpakaNativePtr(device.gainStage),
                                                               alpakaNativePtr(energyBufferResult),
                                                               alpakaNativePtr(clustersBufferResult),
                                                               alpakaNativePtr(usedClusters),
                                                               local_mask,
                                                               device.numMaps,
                                                               i,
                                                               false));

          alpakaEnqueueKernel(device.queue, clusterEnergy);
          alpakaWait(device.queue);	


          //write_img("gain", alpakaNativePtr(gain.data), 0, 3 * sizeof(double));
          //write_img("init_pedestal", alpakaNativePtr(downloadInitialPedestaldata().data), 0, 3 * sizeof(InitPedestal));
          //write_img("pedestal", alpakaNativePtr(downloadPedestaldata().data), 0, 3 * sizeof(Pedestal));
          //write_img("gainstage", alpakaNativePtr(downloadGainStages().data), 16, sizeof(char));
          //write_img("energy", alpakaNativePtr(energy), 16, sizeof(double));
          //write_img("mask2", downloadMask(), 16, sizeof(bool));


          // execute cluster finder
          ClusterFinderKernel<TConfig, TAlpaka> clusterFinderKernel{};
          auto const clusterFinder(alpakaCreateKernel<TAlpaka>(getWorkDiv<TAlpaka>(),
                                                               clusterFinderKernel,
                                                               alpakaNativePtr(dataBuffer.get()),
                                                               alpakaNativePtr(device.gain),
                                                               alpakaNativePtr(device.initialPedestal),
                                                               alpakaNativePtr(device.pedestal),
                                                               alpakaNativePtr(device.gainStage),
                                                               alpakaNativePtr(energyBufferResult),
                                                               alpakaNativePtr(clustersBufferResult),
                                                               alpakaNativePtr(usedClusters),
                                                               local_mask,
                                                               device.numMaps,
                                                               i,
                                                               false));

          alpakaEnqueueKernel(device.queue, clusterFinder);
          alpakaWait(device.queue);
        }

        // download the data
        if (clusters) {
            clustersUsedBuffer.download();

            // wait for completion of copy operations
            alpakaWait(device.queue);
            auto clustersToDownload = alpakaNativePtr(clusters->usedPinned)[0];
            clusters->used += clustersToDownload;

            DEBUG("Downloading ",
                  clustersToDownload,
                  "clusters (",
                  clusters->used,
                  "in total). ");

            clusterBuffer.resize(clustersToDownload);
            clusterBuffer.download();
        }

        energyBuffer.download();
        //write_img("energy", alpakaNativePtr(energy->data), 16, sizeof(double));

        auto wait = [](decltype(device) device) {
            alpakaWait(device.queue);
            return true;
        };

        return std::make_tuple(0, std::async(std::launch::async, []() { return true; }));
    }

private:
    FramePackage<typename TConfig::GainMap, TAlpaka> gain;
    typename TAlpaka::template HostBuf<typename TConfig::MaskMap> mask;
    FramePackage<typename TConfig::GainStageMap, TAlpaka> gainStage;

    FramePackage<typename TConfig::PedestalMap, TAlpaka> pedestal;
    FramePackage<typename TConfig::InitPedestalMap, TAlpaka> initPedestal;

    std::vector<typename TAlpaka::DevAcc> deviceContainer;
  DeviceData<TConfig, TAlpaka> device;

    bool init;    
    double beamConst;

    auto initDevices() -> void
    {
        const GainmapInversionKernel<TConfig> gainmapInversionKernel{};

        alpakaCopy(device.queue,
                   device.gain,
                   gain.data,
                   decltype(TConfig::GAINMAPS)(TConfig::GAINMAPS));
            
        // compute reciprocals of gain maps
        auto const gainmapInversion(alpakaCreateKernel<TAlpaka>(getWorkDiv<TAlpaka>(),
                                                                gainmapInversionKernel,
                                                                alpakaNativePtr(device.gain)));
        alpakaEnqueueKernel(device.queue, gainmapInversion);

        // init cluster data to 0
        alpakaMemSet(device.queue,
                     device.cluster,
                     0,
                     decltype(TConfig::MAX_CLUSTER_NUM_USER)(TConfig::MAX_CLUSTER_NUM_USER) *
                     decltype(TConfig::DEV_FRAMES)(TConfig::DEV_FRAMES));

        // wait until everything is finished
        alpakaWait(device.queue);
            
        DEBUG("Device #", 0, "initialized!");\
    }

    template <typename TDetectorData>
    auto calcPedestaldata(TDetectorData* data, std::size_t numMaps)
        -> std::size_t
    {
        // set the state to processing
        device.numMaps = numMaps;

        // upload the data to the device
        alpakaCopy(device.queue,
                   device.data,
                   alpakaViewPlainPtrHost<TAlpaka, TDetectorData>(
                       data, alpakaGetHost<TAlpaka>(), numMaps),
                   numMaps);
        
        
        // zero out initial pedestal maps and normal pedestal maps
        if (!init) {
            alpakaMemSet(device.queue,
                         device.pedestal,
                         0,
                         decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));
            alpakaMemSet(device.queue,
                         device.initialPedestal,
                         0,
                         decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));
            alpakaWait(device.queue);
            init = true;
        }

        // execute the calibration kernel
        CalibrationKernel<TConfig> calibrationKernel{};
        auto const calibration(
            alpakaCreateKernel<TAlpaka>(getWorkDiv<TAlpaka>(),
                                        calibrationKernel,
                                        alpakaNativePtr(device.data),
                                        alpakaNativePtr(device.initialPedestal),
                                        alpakaNativePtr(device.pedestal),
                                        alpakaNativePtr(device.mask),
                                        device.numMaps));

        alpakaEnqueueKernel(device.queue, calibration);
        alpakaWait(device.queue);

        return numMaps;
    }
};
