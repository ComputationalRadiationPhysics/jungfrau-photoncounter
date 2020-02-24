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
  FramePackage<typename Config::DetectorData, TAccelerator> pedestalData;
  FramePackage<typename Config::DetectorData, TAccelerator> data;
  FramePackage<typename Config::GainMap, TAccelerator> gain;
  double beamConst;
  tl::optional<typename TAccelerator::template HostBuf<MaskMap>> maskPtr;

  // output buffers
  tl::optional<FramePackage<typename Config::EnergyMap, TAccelerator>> energy;
  tl::optional<FramePackage<typename Config::PhotonMap, TAccelerator>> photons;
  tl::optional<FramePackage<typename Config::SumMap, TAccelerator>> sum;
  typename Config::template ClusterArray<TAccelerator> *clusters;
  tl::optional<FramePackage<EnergyValue, TAccelerator>> maxValues;
  ExecutionFlags ef;

  // constructor
  BenchmarkingInput(
      FramePackage<typename Config::DetectorData, TAccelerator> pedestalData,
      FramePackage<typename Config::DetectorData, TAccelerator> data,
      FramePackage<typename Config::GainMap, TAccelerator> gain,
      double beamConst,
      tl::optional<typename TAccelerator::template HostBuf<MaskMap>> maskPtr,
      tl::optional<FramePackage<typename Config::EnergyMap, TAccelerator>>
          energy,
      tl::optional<FramePackage<typename Config::PhotonMap, TAccelerator>>
          photons,
      tl::optional<FramePackage<typename Config::SumMap, TAccelerator>> sum,
      typename Config::template ClusterArray<TAccelerator> *clusters,
      tl::optional<FramePackage<EnergyValue, TAccelerator>> maxValues,
      ExecutionFlags ef)
      : pedestalData(pedestalData), data(data), gain(gain),
        beamConst(beamConst), maskPtr(maskPtr), energy(energy),
        photons(photons), sum(sum), clusters(clusters), maxValues(maxValues),
        ef(ef) {}
};

// prepare and load data for the benchmark
template <typename Config, typename ConcreteAcc>
auto setUp(ExecutionFlags flags, std::string pedestalPath, std::string gainPath,
           std::string dataPath, double beamConst, std::string maskPath = "",
           std::size_t cacheSize = 1024UL * 1024 * 1024 * 16,
           std::size_t maxClusterCount = Config::MAX_CLUSTER_NUM_USER)
    -> BenchmarkingInput<Config, ConcreteAcc> {
  t = Clock::now();

  // create a file cache for all input files
  Filecache<Config> fc(cacheSize);
  DEBUG("filecache created");

  // load maps
  FramePackage<typename Config::DetectorData, ConcreteAcc> pedestalData(
      fc.template loadMaps<typename Config::DetectorData, ConcreteAcc>(
          pedestalPath, true));
  DEBUG(pedestalData.numFrames, "pedestaldata maps loaded");

  FramePackage<typename Config::DetectorData, ConcreteAcc> data(
      fc.template loadMaps<typename Config::DetectorData, ConcreteAcc>(dataPath,
                                                                       true));
  DEBUG(data.numFrames, "data maps loaded");

  FramePackage<typename Config::GainMap, ConcreteAcc> gain(
      fc.template loadMaps<typename Config::GainMap, ConcreteAcc>(gainPath));
  DEBUG(gain.numFrames, "gain maps loaded");

  // create empty, optional input mask
  FramePackage<typename Config::MaskMap, ConcreteAcc> mask(Config::SINGLEMAP);
  mask.numFrames = 0;

  if (maskPath != "") {
    mask =
        fc.template loadMaps<typename Config::MaskMap, ConcreteAcc>(maskPath);
    DEBUG(mask.numFrames, "mask maps loaded");
  }
  
  // create empty, optional input mask
  using MaskMap = typename Config::MaskMap;
  tl::optional<typename ConcreteAcc::template HostBuf<MaskMap>> maskPtr;
  if (mask.numFrames == Config::SINGLEMAP)
    maskPtr = mask.data;
  
  // allocate space for output data
  FramePackage<typename Config::EnergyMap, ConcreteAcc> energy_data(
      data.numFrames);
  FramePackage<typename Config::PhotonMap, ConcreteAcc> photon_data(
      data.numFrames);
  FramePackage<typename Config::SumMap, ConcreteAcc> sum_data(
      (data.numFrames + Config::SUM_FRAMES - 1) / Config::SUM_FRAMES);
  FramePackage<EnergyValue, ConcreteAcc> maxValues_data(data.numFrames);

  // create optional values
  tl::optional<FramePackage<typename Config::EnergyMap, ConcreteAcc>> energy;
  tl::optional<FramePackage<typename Config::PhotonMap, ConcreteAcc>> photon;
  tl::optional<FramePackage<typename Config::SumMap, ConcreteAcc>> sum;
  typename Config::template ClusterArray<ConcreteAcc> *clusters = nullptr;
  tl::optional<FramePackage<EnergyValue, ConcreteAcc>> maxValues;

  // set optional values according to supplied flags
  if (flags.mode == 0) {
    energy = energy_data;
  } else if (flags.mode == 1) {
    photon = photon_data;
  } else if (flags.mode == 2) {
    clusters = new typename Config::template ClusterArray<ConcreteAcc>(
        maxClusterCount * data.numFrames);
  } else {
    energy = energy_data;
    clusters = new typename Config::template ClusterArray<ConcreteAcc>(
        maxClusterCount * data.numFrames);
  }

  if (flags.summation)
    sum = sum_data;
  if (flags.maxValue)
    maxValues = maxValues_data;
  
  // return configuration
  return BenchmarkingInput<Config, ConcreteAcc>(
      pedestalData, data, gain, beamConst, maskPtr, energy, photon, sum,
      clusters, maxValues, flags);
}

// calibrate the detector
template <typename Config, template <std::size_t> typename Accelerator>
auto calibrate(const BenchmarkingInput<Config, Accelerator<Config::MAPSIZE>>
                   &benchmarkingConfig) -> Dispenser<Config, Accelerator> {  
  Dispenser<Config, Accelerator> dispenser(benchmarkingConfig.gain,
                                           benchmarkingConfig.beamConst,
                                           benchmarkingConfig.maskPtr);

  // reset dispenser to get rid of artefacts from previous runs
  dispenser.reset();
  // upload and calculate pedestal data
  dispenser.uploadPedestaldata(benchmarkingConfig.pedestalData);
  dispenser.synchronize();

  return dispenser;
}

// main part for benchmarking
template <typename Config, template <std::size_t> typename Accelerator>
auto bench(
    Dispenser<Config, Accelerator> &dispenser,
    BenchmarkingInput<Config, Accelerator<Config::MAPSIZE>> &benchmarkingConfig)
    -> void {
  using ConcreteAcc = Accelerator<Config::MAPSIZE>;
  using EnergyPackageView =
      FramePackageView_t<typename Config::EnergyMap, ConcreteAcc>;
  using PhotonPackageView =
      FramePackageView_t<typename Config::PhotonMap, ConcreteAcc>;
  using SumPackageView =
      FramePackageView_t<typename Config::SumMap, ConcreteAcc>;
  using MaxValuePackageView = FramePackageView_t<EnergyValue, ConcreteAcc>;

  std::size_t offset = 0;
  std::size_t sum_offset = 0;
  std::vector<std::tuple<std::size_t, std::future<bool>>> futures;

  typename Config::template ClusterArray<ConcreteAcc> *clusters =
      benchmarkingConfig.clusters;





  //! @todo: remove debugging code
  // write out initial pedestal data
  save_image<Config>(
                     "pedestal_start",
                     alpakaNativePtr(dispenser.downloadPedestaldata().data), 0);





  
  // process data maps
  while (offset < benchmarkingConfig.data.numFrames) {

    // define views
    auto energy([&]() -> tl::optional<EnergyPackageView> {
      if (benchmarkingConfig.energy)
        return benchmarkingConfig.energy->getView(offset, Config::DEV_FRAMES);
      return tl::nullopt;
    }());
    auto photons([&]() -> tl::optional<PhotonPackageView> {
      if (benchmarkingConfig.photons)
        return benchmarkingConfig.photons->getView(offset, Config::DEV_FRAMES);
      return tl::nullopt;
    }());
    auto sum([&]() -> tl::optional<SumPackageView> {
      if (benchmarkingConfig.sum)
        return benchmarkingConfig.sum->getView(sum_offset, Config::SUM_FRAMES);
      return tl::nullopt;
    }());
    auto maxValues([&]() -> tl::optional<MaxValuePackageView> {
      if (benchmarkingConfig.maxValues)
        return benchmarkingConfig.maxValues->getView(offset,
                                                     Config::DEV_FRAMES);
      return tl::nullopt;
    }());

    // process data and store results
    futures.emplace_back(dispenser.process(benchmarkingConfig.data, offset,
                                           benchmarkingConfig.ef, energy,
                                           photons, sum, maxValues, clusters));








    //! @todo: remove debugging code
    // write out pedestal and image values
            std::get<1>(futures.back()).wait();
        dispenser.synchronize();

        save_image<Config>("pedestal_" + std::to_string(offset),
            alpakaNativePtr(dispenser.downloadPedestaldata().data), 0);

        if (energy)
          save_image<Config>("energy_" + std::to_string(offset),
                             alpakaNativePtr(energy->data), 0);






        
    auto offset_diff = std::get<0>(*futures.rbegin()) - offset;
    offset = std::get<0>(*futures.rbegin());
    sum_offset += (offset_diff + Config::SUM_FRAMES - 1) / Config::SUM_FRAMES;

    DEBUG(offset, "/", benchmarkingConfig.data.numFrames, "enqueued");
  }

  dispenser.synchronize();
}
