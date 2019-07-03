#pragma once

#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"
#include "jungfrau-photoncounter/Dispenser.hpp"

#include "Filecache.hpp"
#include "jungfrau-photoncounter/Debug.hpp"

// a struct to hold the input parameters for the benchmarking function
template <typename Config, typename TAccelerator> struct BenchmarkingInput {
    using MaskMap = typename Config::MaskMap;

    // input data
    typename Config::template FramePackage<typename Config::DetectorData,
                                           TAccelerator>
        pedestalData;
    typename Config::template FramePackage<typename Config::DetectorData,
                                           TAccelerator>
        data;
    typename Config::template FramePackage<typename Config::GainMap,
                                           TAccelerator>
        gain;
    boost::optional<typename TAccelerator::template HostBuf<MaskMap>> maskPtr;

    // output buffers
    boost::optional<
        typename Config::template FramePackage<typename Config::EnergyMap,
                                               TAccelerator>>
        energy;
    boost::optional<
        typename Config::template FramePackage<typename Config::PhotonMap,
                                               TAccelerator>>
        photons;
    boost::optional<
        typename Config::template FramePackage<typename Config::SumMap,
                                               TAccelerator>>
        sum;
    typename Config::template ClusterArray<TAccelerator>* clusters;
    boost::optional<
        typename Config::template FramePackage<typename Config::EnergyValue,
                                               TAccelerator>>
        maxValues;
    typename Config::ExecutionFlags ef;

    // constructor
    BenchmarkingInput(
        typename Config::template FramePackage<typename Config::DetectorData,
                                               TAccelerator> pedestalData,
        typename Config::template FramePackage<typename Config::DetectorData,
                                               TAccelerator> data,
        typename Config::template FramePackage<typename Config::GainMap,
                                               TAccelerator> gain,
        boost::optional<typename TAccelerator::template HostBuf<MaskMap>>
            maskPtr,
        boost::optional<
            typename Config::template FramePackage<typename Config::EnergyMap,
                                                   TAccelerator>> energy,
        boost::optional<
            typename Config::template FramePackage<typename Config::PhotonMap,
                                                   TAccelerator>> photons,
        boost::optional<
            typename Config::template FramePackage<typename Config::SumMap,
                                                   TAccelerator>> sum,
        typename Config::template ClusterArray<TAccelerator>* clusters,
        boost::optional<
            typename Config::template FramePackage<typename Config::EnergyValue,
                                                   TAccelerator>> maxValues,
        typename Config::ExecutionFlags ef)
        : pedestalData(pedestalData),
          data(data),
          gain(gain),
          maskPtr(maskPtr),
          energy(energy),
          photons(photons),
          sum(sum),
          clusters(clusters),
          maxValues(maxValues),
          ef(ef)
    {
    }
};

// prepare and load data for the benchmark
template <typename Config, typename ConcreteAcc>
auto SetUp(typename Config::ExecutionFlags flags,
           std::string pedestalPath,
           std::string gainPath,
           std::string dataPath,
           std::string maskPath = "",
           std::size_t cacheSize = 1024UL * 1024 * 1024 * 16,
           std::size_t maxClusterCount = Config::MAX_CLUSTER_NUM)
    -> BenchmarkingInput<Config, ConcreteAcc>
{
    t = Clock::now();

    // create a file cache for all input files
    Filecache<Config> fc(cacheSize);
    DEBUG("filecache created");

    // load maps
    typename Config::template FramePackage<typename Config::DetectorData,
                                           ConcreteAcc>
        pedestalData(
            fc.template loadMaps<typename Config::DetectorData, ConcreteAcc>(
                pedestalPath, true));
    DEBUG(pedestalData.numFrames, "pedestaldata maps loaded");

    typename Config::template FramePackage<typename Config::DetectorData,
                                           ConcreteAcc>
        data(fc.template loadMaps<typename Config::DetectorData, ConcreteAcc>(
            dataPath, true));
    DEBUG(data.numFrames, "data maps loaded");

    typename Config::template FramePackage<typename Config::GainMap,
                                           ConcreteAcc>
        gain(fc.template loadMaps<typename Config::GainMap, ConcreteAcc>(
            gainPath));
    DEBUG(gain.numFrames, "gain maps loaded");

    // create empty, optional input mask
    typename Config::template FramePackage<typename Config::MaskMap,
                                           ConcreteAcc>
        mask(Config::SINGLEMAP);
    mask.numFrames = 0;

    if (maskPath != "") {
        mask = fc.template loadMaps<typename Config::MaskMap, ConcreteAcc>(
            maskPath);
        DEBUG(mask.numFrames, "mask maps loaded");
    }

    // create empty, optional input mask
    using MaskMap = typename Config::MaskMap;
    boost::optional<typename ConcreteAcc::template HostBuf<MaskMap>> maskPtr;
    if (mask.numFrames == Config::SINGLEMAP)
        maskPtr = mask.data;

    // allocate space for output data
    typename Config::template FramePackage<typename Config::EnergyMap,
                                           ConcreteAcc>
        energy_data(data.numFrames);
    typename Config::template FramePackage<typename Config::PhotonMap,
                                           ConcreteAcc>
        photon_data(data.numFrames);
    typename Config::template FramePackage<typename Config::SumMap, ConcreteAcc>
        sum_data(data.numFrames);
    typename Config::template FramePackage<typename Config::EnergyValue,
                                           ConcreteAcc>
        maxValues_data(data.numFrames);

    // create optional values
    boost::optional<
        typename Config::template FramePackage<typename Config::EnergyMap,
                                               ConcreteAcc>>
        energy;
    boost::optional<
        typename Config::template FramePackage<typename Config::PhotonMap,
                                               ConcreteAcc>>
        photon;
    boost::optional<
        typename Config::template FramePackage<typename Config::SumMap,
                                               ConcreteAcc>>
        sum;
    typename Config::template ClusterArray<ConcreteAcc>* clusters = nullptr;
    boost::optional<
        typename Config::template FramePackage<typename Config::EnergyValue,
                                               ConcreteAcc>>
        maxValues;

    // set optional values according to supplied flags
    if (flags.mode == 0) {
        energy = energy_data;
    }
    else if (flags.mode == 1) {
        photon = photon_data;
    }
    else if (flags.mode == 2) {
        clusters = new typename Config::template ClusterArray<ConcreteAcc>(
            maxClusterCount * Config::DEV_FRAMES);
    }
    else {
        energy = energy_data;
        clusters = new typename Config::template ClusterArray<ConcreteAcc>(
            maxClusterCount * Config::DEV_FRAMES);
    }

    if (flags.summation)
        sum = sum_data;
    if (flags.maxValue)
        maxValues = maxValues_data;

    // return configuration
    return BenchmarkingInput<Config, ConcreteAcc>(pedestalData,
                                                  data,
                                                  gain,
                                                  maskPtr,
                                                  energy,
                                                  photon,
                                                  sum,
                                                  clusters,
                                                  maxValues,
                                                  flags);
}

// calibrate the detector
template <typename Config, template <std::size_t> typename Accelerator>
auto calibrate(const BenchmarkingInput<Config, Accelerator<Config::MAPSIZE>>&
                   benchmarkingConfig) -> Dispenser<Config, Accelerator>
{
    Dispenser<Config, Accelerator> dispenser(benchmarkingConfig.gain,
                                             benchmarkingConfig.maskPtr);

    // upload and calculate pedestal data
    dispenser.uploadPedestaldata(benchmarkingConfig.pedestalData);
    dispenser.synchronize();

    return dispenser;
}

// main part for benchmarking
template <typename Config, template <std::size_t> typename Accelerator>
auto bench(
    Dispenser<Config, Accelerator>& dispenser,
    BenchmarkingInput<Config, Accelerator<Config::MAPSIZE>>& benchmarkingConfig)
    -> void
{
    using ConcreteAcc = Accelerator<Config::MAPSIZE>;
    using EnergyPackageView =
        typename Config::template FramePackageView_t<typename Config::EnergyMap,
                                                     ConcreteAcc>;
    using PhotonPackageView =
        typename Config::template FramePackageView_t<typename Config::PhotonMap,
                                                     ConcreteAcc>;
    using SumPackageView =
        typename Config::template FramePackageView_t<typename Config::SumMap,
                                                     ConcreteAcc>;
    using MaxValuePackageView = typename Config::
        template FramePackageView_t<typename Config::EnergyValue, ConcreteAcc>;

    std::size_t offset = 0;
    std::size_t downloaded = 0;
    std::size_t currently_downloaded_frames = 0;
    std::vector<std::tuple<std::size_t, std::future<bool>>> uploadFutures;
    std::vector<std::tuple<std::size_t, std::future<bool>>> downloadFutures;

    typename Config::template ClusterArray<ConcreteAcc>* clusters =
        benchmarkingConfig.clusters;

    // process data maps
    while (downloaded < benchmarkingConfig.data.numFrames) {
        uploadFutures.emplace_back(dispenser.uploadData(
            benchmarkingConfig.data, offset, benchmarkingConfig.ef));
        offset = std::get<0>(*uploadFutures.rbegin());

        auto energy([&]() -> boost::optional<EnergyPackageView> {
            if (benchmarkingConfig.energy)
                return benchmarkingConfig.energy->getView(downloaded,
                                                          Config::DEV_FRAMES);
            return boost::none;
        }());
        auto photons([&]() -> boost::optional<PhotonPackageView> {
            if (benchmarkingConfig.photons)
                return benchmarkingConfig.photons->getView(downloaded,
                                                           Config::DEV_FRAMES);
            return boost::none;
        }());
        auto sum([&]() -> boost::optional<SumPackageView> {
            if (benchmarkingConfig.sum)
                return benchmarkingConfig.sum->getView(downloaded,
                                                       Config::SUM_FRAMES);
            return boost::none;
        }());
        auto maxValues([&]() -> boost::optional<MaxValuePackageView> {
            if (benchmarkingConfig.maxValues)
                return benchmarkingConfig.maxValues->getView(
                    downloaded, Config::DEV_FRAMES);
            return boost::none;
        }());

        downloadFutures.emplace_back(
            dispenser.downloadData(energy, photons, sum, maxValues, clusters));
        currently_downloaded_frames = std::get<0>(*downloadFutures.rbegin());
        if (currently_downloaded_frames) {
            downloaded += currently_downloaded_frames;
            DEBUG(downloaded,
                  "/",
                  benchmarkingConfig.data.numFrames,
                  "downloaded;",
                  offset,
                  "uploaded");
        }
    }

    dispenser.synchronize();
}
