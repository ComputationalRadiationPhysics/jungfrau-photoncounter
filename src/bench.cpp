#include "jungfrau-photoncounter/Dispenser.hpp"

#include <chrono>
#include <vector>
#include <fstream>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */

#if defined (ALPAKA_ACC_GPU_CUDA_ENABLED)
const char* dvstr = "CUDA";
template <std::size_t TMapSize> using Accelerator = GpuCudaRt<TMapSize>;
#elif defined (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
const char* dvstr = "OMP";
template <std::size_t TMapSize> using Accelerator = CpuOmp2Blocks<TMapSize>;
#else
const char* dvstr = "Serial";
template <std::size_t TMapSize> using Accelerator = CpuSerial<TMapSize>;
#endif

// CpuOmp2Blocks<MAPSIZE>;
// CpuTbbRt<MAPSIZE>;
// CpuSerial<MAPSIZE>;
// GpuCudaRt<MAPSIZE>;
// GpuHipRt<MAPSIZE>;

using Duration = std::chrono::nanoseconds;
using Timer = std::chrono::high_resolution_clock;

template <typename TData, typename TAlpaka>
auto loadMaps(const std::string &path)
      -> FramePackage<TData, TAlpaka> {
    // get file size
    struct stat fileStat;
    stat(path.c_str(), &fileStat);
     auto fileSize = fileStat.st_size;
    std::size_t numFrames = fileSize / sizeof(TData);

    // check for empty files
    if (fileSize == 0) {
      std::cerr << "Error: Nothing loaded! Is the file path correct?\n";
      exit(EXIT_FAILURE);
    }

    // allocate space for data
    FramePackage<TData, TAlpaka> maps(numFrames);

    // load file content
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Couldn't open file " << path << "!\n";
      exit(EXIT_FAILURE);
    }

    file.read(reinterpret_cast<char *>(alpakaNativePtr(maps.data)), fileSize);
    file.close();
    
    return maps;
  }


void benchmark(unsigned int iterations,
                                ExecutionFlags flags,
                                const std::string& pedestalPath,
                                const std::string& gainPath,
                                const std::string& dataPath,
                                double beamConst)
{
    //using Config = typename ConfigFrom<Tuple>::Result;
    using Config = DetectorConfig<1000, 1000, 999, 1024, 512, 2, 1, 100, 7, 5>;
    using ConcreteAcc = Accelerator<Config::MAPSIZE>;

    std::cout << "Parameters: sumFrames=" << Config::SUM_FRAMES
              << "; devFrames=" << Config::DEV_FRAMES
              << "; clusterSize=" << Config::CLUSTER_SIZE << "\n";
    std::cout << "Flags: mode=" << static_cast<int>(flags.mode)
              << "; summation=" << static_cast<int>(flags.summation)
              << "; masking=" << static_cast<int>(flags.masking)
              << "; maxValue=" << static_cast<int>(flags.maxValue) << "\n";

    std::string maskPath = "mask.dat";
    std::size_t maxClusterCount = Config::MAX_CLUSTER_NUM_USER;

    t = Clock::now();

    // load maps
    FramePackage<typename Config::DetectorData, ConcreteAcc> pedestalData(loadMaps<typename Config::DetectorData, ConcreteAcc>(
                                                                                                                               pedestalPath));
    DEBUG(pedestalData.numFrames, "pedestaldata maps loaded");

    FramePackage<typename Config::DetectorData, ConcreteAcc> data(loadMaps<typename Config::DetectorData, ConcreteAcc>(dataPath));
    DEBUG(data.numFrames, "data maps loaded");

    FramePackage<typename Config::GainMap, ConcreteAcc> gain(loadMaps<typename Config::GainMap, ConcreteAcc>(gainPath));
    DEBUG(gain.numFrames, "gain maps loaded");

    // create empty, optional input mask
    FramePackage<typename Config::MaskMap, ConcreteAcc> mask(Config::SINGLEMAP);
    mask.numFrames = 0;

    if (maskPath != "") {
      mask =
        loadMaps<typename Config::MaskMap, ConcreteAcc>(maskPath);
      DEBUG(mask.numFrames, "mask maps loaded");
    }
  
    // create empty, optional input mask
    using MaskMap = typename Config::MaskMap;
    tl::optional<typename ConcreteAcc::template HostBuf<MaskMap>> maskPtr;
    if (mask.numFrames == Config::SINGLEMAP)
      maskPtr = mask.data;
  
    // allocate space for output data
    FramePackage<typename Config::EnergyMap, ConcreteAcc> energy_data(data.numFrames);

    // create optional values
    tl::optional<FramePackage<typename Config::EnergyMap, ConcreteAcc>> energy;
    typename Config::template ClusterArray<ConcreteAcc> *clusters = nullptr;
  

    energy = energy_data;
    clusters = new typename Config::template ClusterArray<ConcreteAcc>(maxClusterCount * data.numFrames);  
  
    DEBUG("Initialization done!");  

    if (clusters)
      clusters->used = 0;

    unsigned int moduleNumber = 0;
    unsigned int moduleCount = 1;

    FramePackage<typename Config::InitPedestalMap, ConcreteAcc> initialPedestals(loadMaps<typename Config::InitPedestalMap, ConcreteAcc>("init_pedestal.dat"));
    DEBUG(initialPedestals.numFrames, "initial pedestal maps loaded");

    Dispenser<Config, Accelerator> dispenser(gain,
                                             beamConst,
                                             maskPtr, moduleNumber, moduleCount);

    // reset dispenser to get rid of artefacts from previous runs
    dispenser.reset();
    // upload and calculate pedestal data
    //dispenser.uploadRawPedestals(initialPedestals);
    dispenser.uploadPedestaldata(pedestalData);
    dispenser.synchronize();

    using EnergyPackageView =
      FramePackageView_t<typename Config::EnergyMap, ConcreteAcc>;
    using PhotonPackageView =
      FramePackageView_t<typename Config::PhotonMap, ConcreteAcc>;
    using SumPackageView =
      FramePackageView_t<typename Config::SumMap, ConcreteAcc>;
    using MaxValuePackageView = FramePackageView_t<EnergyValue, ConcreteAcc>;

    std::tuple<std::size_t, std::future<bool>> future;

    // define views
    auto oenergy([&]() -> tl::optional<EnergyPackageView> {
        return energy->getView(0, 1);
      }());
    auto ophotons([&]() -> tl::optional<PhotonPackageView> {
        return tl::nullopt;
      }());
    auto osum([&]() -> tl::optional<SumPackageView> {
        return tl::nullopt;
      }());
    auto omaxValues([&]() -> tl::optional<MaxValuePackageView> {
        return tl::nullopt;
      }());

    // process data and store results
    future = dispenser.process(data, 0, oenergy,
                               ophotons, clusters);

    dispenser.synchronize();
}

int main(int argc, char* argv[])
{
    // check command line parameters
    if (argc < 11 || argc > 17) {
        std::cerr << "Usage: bench <benchmark id> <iteration count> "
                     "<beamConst> <mode> <masking> <max values> <summation> "
                     "<pedestal path> <gain path> <data path> [output prefix] "
                     "[energy reference result path] [photon reference result "
                     "path] [max value reference result path] [sum reference "
                     "result path] [cluster reference result path]\n";
        abort();
    }

    // initialize parameters
    int benchmarkID = std::atoi(argv[1]);
    unsigned int iterationCount = static_cast<unsigned int>(std::atoi(argv[2]));
    double beamConst = std::atof(argv[3]);
    ExecutionFlags ef;
    ef.mode = static_cast<std::uint8_t>(std::atoi(argv[4]));
    ef.masking = static_cast<std::uint8_t>(std::atoi(argv[5]));
    ef.maxValue = static_cast<std::uint8_t>(std::atoi(argv[6]));
    ef.summation = static_cast<std::uint8_t>(std::atoi(argv[7]));

    std::string pedestalPath(argv[8]);
    std::string gainPath(argv[9]);
    std::string dataPath(argv[10]);
    std::string outputPath((argc >= 12) ? std::string(argv[11]) + "_" : "");

    // run benchmark
    //using Config = DetectorConfig<1000, 1000, 999, 1024, 512, 2, 1, 100, 7, 5>;
    //using ConcreteAcc = Accelerator<Config::MAPSIZE>;

    //std::cout << "Parameters: sumFrames=" << Config::SUM_FRAMES
    //          << "; devFrames=" << Config::DEV_FRAMES
    //          << "; clusterSize=" << Config::CLUSTER_SIZE << "\n";
    //std::cout << "Flags: mode=" << static_cast<int>(ef.mode)
    //          << "; summation=" << static_cast<int>(ef.summation)
    //          << "; masking=" << static_cast<int>(ef.masking)
    //          << "; maxValue=" << static_cast<int>(ef.maxValue) << "\n";

    //auto benchmarkingInput = setUp<Config, ConcreteAcc>(
    //    ef, pedestalPath, gainPath, dataPath, beamConst);
    //if (benchmarkingInput.clusters)
    //    benchmarkingInput.clusters->used = 0;
    //auto dispenser = calibrate(benchmarkingInput);
    //auto t0 = Timer::now();
    //bench(dispenser, benchmarkingInput);
    //auto t1 = Timer::now();

    //DEBUG(std::chrono::duration_cast<Duration>(t1 - t0).count());

    benchmark(iterationCount, 
              ef,
              pedestalPath,
              gainPath,
              dataPath,
              beamConst);

    DEBUG(dvstr);

    return 0;
}
