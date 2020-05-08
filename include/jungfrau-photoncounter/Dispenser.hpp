#pragma once

#include "Alpakaconfig.hpp"

#include "kernel/ClusterFinder.hpp"
#include "kernel/GainmapInversion.hpp"

#include <optional.hpp>

#include <iostream>

#include "Config.hpp"

template <typename Config, typename TAlpaka> struct DeviceData {
    typename TAlpaka::DevAcc* device;
    typename TAlpaka::Queue queue;

    // device maps
    typename TAlpaka::template AccBuf<typename Config::DetectorData> data;
    typename TAlpaka::template AccBuf<typename Config::GainMap> gain;
    typename TAlpaka::template AccBuf<typename Config::InitPedestalMap>
        initialPedestal;
    typename TAlpaka::template AccBuf<typename Config::PedestalMap> pedestal;
    typename TAlpaka::template AccBuf<typename Config::MaskMap> mask;
    typename TAlpaka::template AccBuf<typename Config::GainStageMap> gainStage;
    typename TAlpaka::template AccBuf<typename Config::EnergyMap> energy;
    typename TAlpaka::template AccBuf<typename Config::Cluster> cluster;
    typename TAlpaka::template AccBuf<unsigned long long> numClusters;

    DeviceData(typename TAlpaka::DevAcc* devPtr)
        : device(devPtr),
          queue(*device),
          data(alpakaAlloc<typename Config::DetectorData>(
              *device,
              decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
          gain(alpakaAlloc<typename Config::GainMap>(
              *device,
              decltype(Config::GAINMAPS)(Config::GAINMAPS))),
          pedestal(alpakaAlloc<typename Config::PedestalMap>(
              *device,
              decltype(Config::PEDEMAPS)(Config::PEDEMAPS))),
          initialPedestal(alpakaAlloc<typename Config::InitPedestalMap>(
              *device,
              decltype(Config::PEDEMAPS)(Config::PEDEMAPS))),
          gainStage(alpakaAlloc<typename Config::GainStageMap>(
              *device,
              decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
          energy(alpakaAlloc<typename Config::EnergyMap>(
              *device,
              decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
          mask(alpakaAlloc<typename Config::MaskMap>(
              *device,
              decltype(Config::SINGLEMAP)(Config::SINGLEMAP))),
          cluster(alpakaAlloc<typename Config::Cluster>(
              *device,
              decltype(Config::MAX_CLUSTER_NUM_USER)(
                  Config::MAX_CLUSTER_NUM_USER) *
                  decltype(Config::DEV_FRAMES)(Config::DEV_FRAMES))),
          numClusters(alpakaAlloc<unsigned long long>(
              *device,
              decltype(Config::SINGLEMAP)(Config::SINGLEMAP )))
    {
        // set cluster counter to 0
        alpakaMemSet(queue,
                     numClusters,
                     0,
                     decltype(Config::SINGLEMAP)(Config::SINGLEMAP));
    }
};



template <typename TConfig, template <std::size_t> typename TAccelerator>
class Dispenser {
public:
    // use types defined in the config struct
    using TAlpaka = TAccelerator<TConfig::MAPSIZE>;

    Dispenser(FramePackage<typename TConfig::GainMap, TAlpaka> gain,
              FramePackage<typename TConfig::MaskMap, TAlpaka> mask,
	      FramePackage<typename TConfig::InitPedestalMap, TAlpaka> initialPedestals, 
	      FramePackage<typename TConfig::PedestalMap, TAlpaka> pedestals)
        : device(&alpakaGetDevs<TAlpaka>()[0])
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

        DEBUG("Loading existing mask map on device", 0);
        alpakaCopy(device.queue,
                   device.mask,
                   mask.data,
                   decltype(TConfig::SINGLEMAP)(TConfig::SINGLEMAP));

        DEBUG("distribute raw pedestal maps");
        alpakaCopy(device.queue,
                   device.initialPedestal,
                   initialPedestals.data,
                   decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));
        alpakaCopy(device.queue,
                   device.pedestal,
                   pedestals.data,
                   decltype(TConfig::PEDEMAPS)(TConfig::PEDEMAPS));

        alpakaWait(device.queue);

        DEBUG("Device #", 0, "initialized!");
    }

  template <typename TFramePackageEnergyMap>
    auto process(FramePackage<typename TConfig::DetectorData, TAlpaka> data,
                 std::size_t offset,
                 tl::optional<TFramePackageEnergyMap> energy,
                 typename TConfig::template ClusterArray<TAlpaka>* clusters)
        -> void {

      auto numMaps = data.numFrames % (TConfig::DEV_FRAMES + 1);

        using ClusterView =
            typename TAlpaka::template HostView<typename TConfig::Cluster>;

        // create a shadow buffer on the device and upload data if required
        FramePackageDoubleBuffer<TAlpaka,
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
                                                               numMaps,
                                                               i,
                                                               false));

          alpakaEnqueueKernel(device.queue, clusterEnergy);
          alpakaWait(device.queue);	

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
                                                               numMaps,
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

        alpakaWait(device.queue);
    }

private:
  DeviceData<TConfig, TAlpaka> device;
};
