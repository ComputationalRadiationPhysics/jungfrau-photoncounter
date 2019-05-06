//#include <benchmark/benchmark.h>

#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"
#include "jungfrau-photoncounter/Dispenser.hpp"

#include "jungfrau-photoncounter/Debug.hpp"
#include "Filecache.hpp"

// flags for benchmarking
struct BenchmarkingFlags {
    uint8_t downloadEnergy : 1;
    uint8_t downloadPhotons : 1;
    uint8_t downloadSum : 1;
    uint8_t downloadClusters : 1;
    uint8_t downloadMaxValue : 1;
};

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
    typename Config::template FramePackage<typename Config::DetectorData,
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
    boost::optional<typename Config::template ClusterArray<TAccelerator>>
        clusters;
    boost::optional<
        typename Config::template FramePackage<typename Config::EnergyValue,
                                               TAccelerator>>
        maxValues;

    // constructor
    BenchmarkingInput(
        typename Config::template FramePackage<typename Config::DetectorData,
                                               TAccelerator> pedestalData,
        typename Config::template FramePackage<typename Config::DetectorData,
                                               TAccelerator> data,
        typename Config::template FramePackage<typename Config::DetectorData,
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
        boost::optional<typename Config::template ClusterArray<TAccelerator>>
            clusters,
        boost::optional<
            typename Config::template FramePackage<typename Config::EnergyValue,
                                                   TAccelerator>> maxValues)
        : pedestalData(pedestalData),
          data(data),
          gain(gain),
          maskPtr(maskPtr),
          energy(energy),
          photons(photons),
          sum(sum),
          clusters(clusters),
          maxValues(maxValues)
    {
    }
};

// prepare and load data for the benchmark
template <typename Config, typename ConcreteAcc>
auto SetUp(BenchmarkingFlags flags,
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
        pedestaldata(fc.loadMaps<typename Config::DetectorData, ConcreteAcc>(
            pedestalPath, true));
    DEBUG(pedestaldata.numFrames, "pedestaldata maps loaded");

    typename Config::template FramePackage<typename Config::DetectorData,
                                           ConcreteAcc>
        data(fc.loadMaps<typename Config::DetectorData, ConcreteAcc>(dataPath,
                                                                      true));
    DEBUG(data.numFrames, "data maps loaded");

    typename Config::template FramePackage<typename Config::GainMap,
                                           ConcreteAcc>
        gain(fc.loadMaps<typename Config::GainMap, ConcreteAcc>(gainPath));
    DEBUG(gain.numFrames, "gain maps loaded");

    // create empty, optional input mask
    typename Config::template FramePackage<typename Config::MaskMap,
                                           ConcreteAcc>
        mask(Config::SINGLEMAP);
    mask.numFrames = 0;

    if (maskPath != "") {
        mask = fc.loadMaps<typename Config::MaskMap, ConcreteAcc>(maskPath);
        DEBUG(mask.numFrames, "mask maps loaded");
    }

    // create empty, optional input mask
    boost::optional<
        typename ConcreteAcc::template HostBuf<typename Config::MaskMap>>
        maskPtr;
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
    typename Config::template ClusterArray<ConcreteAcc> clusters_data(
        maxClusterCount);
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
    boost::optional<typename Config::template ClusterArray<ConcreteAcc>>
        clusters;
    boost::optional<
        typename Config::template FramePackage<typename Config::EnergyValue,
                                               ConcreteAcc>>
        maxValues;

    // set optional values according to supplied flags
    if (flags.downloadEnergy)
        energy = energy_data;
    if (flags.downloadPhotons)
        photon = photon_data;
    if (flags.downloadSum)
        sum = sum_data;
    if (flags.downloadClusters)
        clusters = cluster_data;
    if (flags.downloadMaxValue)
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
                                                  maxValues);
}

// calibrate the detector
template <typename Config, template <std::size_t> typename Accelerator>
auto calibrate(const BenchmarkingInput<Config, Accelerator<Config::MAPSIZE>>&
                   benchmarkingConfig) -> Dispenser<Config, Accelerator>
{
    Dispenser<Config, Accelerator> dispenser(benchmarkingConfig.gain,
                                             benchmarkingConfig.maskPtr);

    // upload and calculate pedestal data
    dispenser.uploadPedestaldata(benchmarkingConfig.pedestaldata);
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
    std::size_t offset = 0;
    std::size_t downloaded = 0;
    std::size_t currently_downloaded_frames = 0;

    // process data maps
    while (downloaded < benchmarkingConfig.data.numFrames) {
        offset = dispenser.uploadData(benchmarkingConfig.data, offset, ef);
        if (currently_downloaded_frames = dispenser.downloadData(
                benchmarkingConfig.energy.getView(downloaded,
                                                  Config::DEV_FRAMES),
                benchmarkingConfig.photon.getView(downloaded,
                                                  Config::DEV_FRAMES),
                benchmarkingConfig.sum.getView(downloaded, Config::DEV_FRAMES),
                benchmarkingConfig.maxValues.getView(downloaded,
                                                     Config::DEV_FRAMES),
                benchmarkingConfig.clusters)) {
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


/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */
template <std::size_t MAPSIZE>
using Accelerator = GpuCudaRt<MAPSIZE>; // CpuSerial<MAPSIZE>;//GpuCudaRt<MAPSIZE>;//GpuCudaRt<MAPSIZE>;
                                        // // CpuSerial;
using Config = MoenchConfig;
using ConcreteAcc = Accelerator<Config::MAPSIZE>;

auto main(int argc, char* argv[]) -> int
{
    BenchmarkingFlags flags = {0, 0, 0, 1, 0};
    std::string pedestalPath =
        "../../moench_data/1000_frames_pede_e17050_1_00018_00000.dat";
    //"../../data_pool/px_101016/allpede_250us_1243__B_000000.dat"
    std::string gainPath = "../../moench_data/moench_gain.bin";
    //"../../data_pool/px_101016/gainMaps_M022.bin"
    std::string dataPath = "../../moench_data/e17050_1_00018_00000_image.dat";
    //"../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat"

    auto benchmarkingInput =
        SetUp<Config, ConcreteAcc>(flags, pedestalPath, gainPath, dataPath);
    auto dispenser = calibrate<Config, Accelerator>(benchmarkingInput);
    bench<Config, Accelerator>(dispenser, benchmarkingInput);


    // debugging:
    if (benchmarkingInput.clusters) {
        saveClusters("clusters.txt", *benchmarkingInput.clusters);
        saveClustersBin("clusters.bin", *benchmarkingInput.clusters);
    }

    auto sizes = dispenser.getMemSize();
    auto free_mem = dispenser.getFreeMem();

    for (std::size_t i = 0; i < sizes.size(); ++i)
        DEBUG("Device #",
              i,
              ":",
              (float)free_mem[i] / (float)sizes[i] * 100.0f,
              "% free of",
              sizes[i],
              "Bytes");

    return 0;
}
