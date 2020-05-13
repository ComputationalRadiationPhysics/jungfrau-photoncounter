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

template <typename TData, typename TAlpaka>
auto loadMap(const std::string &path)
  -> typename TAlpaka::template HostBuf<TData> {
    // get file size
    struct stat fileStat;
    stat(path.c_str(), &fileStat);
     auto fileSize = fileStat.st_size;

    // check for empty files
    if (fileSize == 0 || fileSize < sizeof(TData)) {
      std::cerr << "Error: Nothing loaded! Is the file path correct (" << path << ")?\n";
      exit(EXIT_FAILURE);
    }

    // allocate space for data
    typename TAlpaka::template HostBuf<TData> map(alpakaAlloc<TData>(alpakaGetHost<TAlpaka>(), (std::size_t)1));

    // load file content
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error: Couldn't open file " << path << "!\n";
      exit(EXIT_FAILURE);
    }

    file.read(reinterpret_cast<char *>(alpakaNativePtr(map)), sizeof(TData));
    file.close();
    
    return map;
  }

int main(int argc, char* argv[])
{    
    std::string gainPath("../../../data_pool/px_101016/gainMaps_M022.bin");
    std::string dataPath("../../../data_pool/px_101016/Insu1.dat");

    using Config = DetectorConfig<1024, 512, 1, 7>;
    using ConcreteAcc = Accelerator<Config::MAPSIZE>;

    std::cout << "Parameters: devFrames=" << Config::DEV_FRAMES
              << "; clusterSize=" << Config::CLUSTER_SIZE << "\n";

    std::string maskPath = "mask.dat";

    t = Clock::now();

    FramePackage<typename Config::DetectorData, ConcreteAcc> data(1);
    data.data = loadMap<typename Config::DetectorData, ConcreteAcc>(dataPath);
    DEBUG(data.numFrames, "data maps loaded");
    
    FramePackage<typename Config::EnergyMap, ConcreteAcc> energydata(1);
    energydata.data = loadMap<typename Config::EnergyMap, ConcreteAcc>("energy.dat");
    DEBUG(data.numFrames, "energy maps loaded");
    
    FramePackage<typename Config::GainStageMap, ConcreteAcc> gainstagedata(1);
    gainstagedata.data = loadMap<typename Config::GainStageMap, ConcreteAcc>("gainstage_raw.dat");
    DEBUG(data.numFrames, "gainstage maps loaded");

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

    DEBUG("Initialization done!");  

    FramePackage<typename Config::InitPedestalMap, ConcreteAcc> initialPedestals(loadMaps<typename Config::InitPedestalMap, ConcreteAcc>("init_pedestal.dat"));
    DEBUG(initialPedestals.numFrames, "initial pedestal maps loaded");

    FramePackage<typename Config::PedestalMap, ConcreteAcc> pedestals(loadMaps<typename Config::PedestalMap, ConcreteAcc>("pedestal.dat"));
    DEBUG(initialPedestals.numFrames, "pedestal maps loaded");

    Dispenser<Config, Accelerator> dispenser(gain, mask, initialPedestals, pedestals);

    // process data and store results
    dispenser.process(loadMap<typename Config::DetectorData, ConcreteAcc>(dataPath), 0, energy_data, energydata, gainstagedata);

    DEBUG(dvstr);

    return 0;
}
