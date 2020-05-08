#include "jungfrau-photoncounter/Dispenser.hpp"

#include <chrono>
#include <vector>
#include <fstream>
#include <memory>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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
      std::cerr << "Error: Nothing loaded! Is the file path correct (" << path << ")?\n";
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

int main(int argc, char* argv[])
{    
    std::string gainPath("../../../data_pool/px_101016/gainMaps_M022.bin");
    std::string dataPath("../../../data_pool/px_101016/Insu1.dat");

    using Config = DetectorConfig<1000, 1000, 999, 1024, 512, 2, 1, 100, 7, 5>;
    using ConcreteAcc = Accelerator<Config::MAPSIZE>;

    std::cout << "Parameters: sumFrames=" << Config::SUM_FRAMES
              << "; devFrames=" << Config::DEV_FRAMES
              << "; clusterSize=" << Config::CLUSTER_SIZE << "\n";

    std::string maskPath = "mask.dat";
    std::size_t maxClusterCount = Config::MAX_CLUSTER_NUM_USER;

    t = Clock::now();

    FramePackage<typename Config::DetectorData, ConcreteAcc> data(loadMaps<typename Config::DetectorData, ConcreteAcc>(dataPath));
    DEBUG(data.numFrames, "data maps loaded");

    FramePackage<typename Config::GainMap, ConcreteAcc> gain(loadMaps<typename Config::GainMap, ConcreteAcc>(gainPath));
    DEBUG(gainPath);
    DEBUG(gain.numFrames, "gain maps loaded");

    FramePackage<typename Config::GainMap, ConcreteAcc> raw_gain(loadMaps<typename Config::GainMap, ConcreteAcc>("gain.dat"));
    DEBUG(raw_gain.numFrames, "gain maps 2 loaded");

    // create empty, optional input mask
    FramePackage<typename Config::MaskMap, ConcreteAcc> mask(Config::SINGLEMAP);
    mask.numFrames = 0;

    if (maskPath != "") {
      mask =
        loadMaps<typename Config::MaskMap, ConcreteAcc>(maskPath);
      DEBUG(mask.numFrames, "mask maps loaded");
    }
  
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

    FramePackage<typename Config::InitPedestalMap, ConcreteAcc> initialPedestals(loadMaps<typename Config::InitPedestalMap, ConcreteAcc>("init_pedestal.dat"));
    DEBUG(initialPedestals.numFrames, "initial pedestal maps loaded");

    FramePackage<typename Config::PedestalMap, ConcreteAcc> pedestals(loadMaps<typename Config::PedestalMap, ConcreteAcc>("pedestal.dat"));
    DEBUG(initialPedestals.numFrames, "pedestal maps loaded");

    Dispenser<Config, Accelerator> dispenser(gain, mask, initialPedestals, pedestals);

    using EnergyPackageView =
      FramePackageView_t<typename Config::EnergyMap, ConcreteAcc>;

    // define views
    auto oenergy([&]() -> tl::optional<EnergyPackageView> {
        return energy->getView(0, 1);
      }());

    // process data and store results
    dispenser.process(data, 0, oenergy, clusters);

    DEBUG(dvstr);

    return 0;
}
