#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"
#include "jungfrau-photoncounter/Dispenser.hpp"

#include "Filecache.hpp"

#include "jungfrau-photoncounter/Debug.hpp"

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template <std::size_t TMapSize> using Accelerator = GpuCudaRt<TMapSize>;
#else
template <std::size_t TMapSize> using Accelerator = CpuSerial<TMapSize>;
#endif
// CpuOmp2Blocks<MAPSIZE>;
// CpuTbbRt<MAPSIZE>;
// CpuSerial<MAPSIZE>;
// GpuCudaRt<MAPSIZE>;
// GpuHipRt<MAPSIZE>;

//#define MOENCH
#define SYNTH8

#ifdef MOENCH
using Config = MoenchConfig;
constexpr float BeamConst = 8.7f;
const std::string pedestalPath =
    "../../../moench_data/1000_frames_pede_e17050_1_00018_00000.dat";
const std::string gainPath = "../../../moench_data/moench_gain.bin";
const std::string dataPath =
    "../../../moench_data/e17050_1_00018_00000_image.dat";
//#else//if SYNTH8
//using Config = JungfrauConfig;
//constexpr float BeamConst = 12.4f;
//const std::string pedestalPath =
//    "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/pede.bin";
//const std::string gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin";
//const std::string dataPath =
//    "/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_8_100.bin";
//"/bigdata/hplsim/production/jungfrau-photoncounter/data_pool/synthetic/cluster_8.bin";
#else
using Config = JungfrauConfig;
constexpr float BeamConst = 12.4f;
const std::string pedestalPath =
    "../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat";
const std::string gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin";
const std::string dataPath =
  "../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000_3.dat";
//"../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat";
#endif

using ConcreteAcc = Accelerator<Config::MAPSIZE>;

auto main(int argc, char* argv[]) -> int
{
    // t is used in all debug-messages
    t = Clock::now();
    
    // create a file cache for all input files
    Filecache<Config>* fc = new Filecache<Config>(1024UL * 1024 * 1024 * 16);
    DEBUG("filecache created");

    // load maps
    FramePackage<typename Config::DetectorData, ConcreteAcc> pedestaldata(
        fc->loadMaps<typename Config::DetectorData, ConcreteAcc>(pedestalPath, true));
    DEBUG(pedestaldata.numFrames, "pedestaldata maps loaded from", pedestalPath);

    FramePackage<typename Config::DetectorData, ConcreteAcc> data(
        fc->loadMaps<typename Config::DetectorData, ConcreteAcc>(dataPath, true));
    DEBUG(data.numFrames, "data maps loaded from", dataPath);

    FramePackage<typename Config::GainMap, ConcreteAcc> gain(
        fc->loadMaps<typename Config::GainMap, ConcreteAcc>(gainPath));
    DEBUG(gain.numFrames, "gain maps loaded from", gainPath);

    float beamConst = BeamConst;
    // Jungfrau: 12.4keV
    // MOENCH: 8.7keV

    FramePackage<typename Config::MaskMap, ConcreteAcc> mask(Config::SINGLEMAP);
    mask.numFrames = 0;
    delete (fc);

    // create empty, optional input mask
    tl::optional<ConcreteAcc::HostBuf<typename Config::MaskMap>> maskPtr;
    if (mask.numFrames == Config::SINGLEMAP)
        maskPtr = mask.data;

    // initialize the dispenser
    Dispenser<Config, Accelerator> dispenser(gain, beamConst, maskPtr);

    // print info
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    DEBUG("gpu count:", dispenser.getMemSize().size());
#endif

    // upload and calculate pedestal data
   dispenser.uploadPedestaldata(pedestaldata);

    // allocate space for output data
    FramePackage<typename Config::EnergyMap, ConcreteAcc> energy_data(
        data.numFrames);
    FramePackage<typename Config::PhotonMap, ConcreteAcc> photon_data(
        data.numFrames);
    // note: the last term allocates space for at least one additional map for
    // each package to count for the accumulated rounding up in the main loop
    FramePackage<typename Config::SumMap, ConcreteAcc> sum_data(
        (data.numFrames + Config::SUM_FRAMES - 1) / Config::SUM_FRAMES +
        (data.numFrames + Config::DEV_FRAMES - 1) / Config::DEV_FRAMES);
    typename Config::ClusterArray<ConcreteAcc> clusters_data(Config::MAX_CLUSTER_NUM_USER * data.numFrames);
    FramePackage<EnergyValue, ConcreteAcc> maxValues_data(data.numFrames);

    tl::optional<FramePackage<typename Config::EnergyMap, ConcreteAcc>> energy =
        energy_data;
    tl::optional<FramePackage<typename Config::PhotonMap, ConcreteAcc>> photon =
        photon_data;
    tl::optional<FramePackage<typename Config::SumMap, ConcreteAcc>> sum =
        sum_data;
    typename Config::ClusterArray<ConcreteAcc>* clusters = &clusters_data;

    tl::optional<FramePackage<EnergyValue, ConcreteAcc>> maxValues =
        maxValues_data;

    // create variable to track the progress
    std::size_t offset = 0;

    // create vectors to hold the number of downloaded / uploaded frames and a
    // future The future is a variable, which is not "ready", when it is
    // returned from a function. Internally, this contains a set of instructions
    // to calculate the result. So if a future is accessed, the program waits
    // until the variable is processed. We use the future to signalize that an
    // upload or download operation is done. The internal value, which is then
    // returned, doesn't hold any deeper meaning. Once a future goes out pf
    // scope, it waits for its variable to be calculated. To delay this to the
    // end of the program, we create a vector of futures and store them there.
    std::vector<std::tuple<std::size_t, std::future<bool>>> futures;

    // set execution flags
    ExecutionFlags ef;
    ef.mode = 0;
    ef.summation = 0;
    ef.masking = 1;
    ef.maxValue = 0;

    using EnergyPackageView =
        FramePackageView_t<typename Config::EnergyMap, ConcreteAcc>;
    using PhotonPackageView =
        FramePackageView_t<typename Config::PhotonMap, ConcreteAcc>;
    using SumPackageView =
        FramePackageView_t<typename Config::SumMap, ConcreteAcc>;
    using MaxValuePackageView = FramePackageView_t<EnergyValue, ConcreteAcc>;




    save_image<Config>("mask",
                       dispenser.downloadMask(), 0);


    save_image<Config>(
        "pedestal", alpakaNativePtr(dispenser.downloadPedestaldata().data), 0);
    
  // download and save std dev of the initial pedestal map                                                                                                                                                                                                   
  auto initPed = dispenser.downloadInitialPedestaldata();
  std::ofstream outPede("pede_stddev.bin", std::ios::binary);
  InitPedestal *pedePtr = alpakaNativePtr(initPed.data)->data;
  for (unsigned int y = 0; y < Config::DIMY; ++y)
    for (unsigned int x = 0; x < Config::DIMX; ++x)
      outPede.write(reinterpret_cast<char *>(&(pedePtr[y * 1024 + x].stddev)),
                    sizeof(double));
  outPede.flush();
  outPede.close();

  outPede.open("pede_m.bin", std::ios::binary);
  pedePtr = alpakaNativePtr(initPed.data)->data;
  for (unsigned int y = 0; y < Config::DIMY; ++y)
    for (unsigned int x = 0; x < Config::DIMX; ++x)
      outPede.write(
          reinterpret_cast<char *>(&(pedePtr[y * Config::DIMX + x].m)),
          sizeof(pedePtr[y * Config::DIMX + x].m));
  outPede.flush();
  outPede.close();

  outPede.open("pede_m2.bin", std::ios::binary);
  pedePtr = alpakaNativePtr(initPed.data)->data;
  for (unsigned int y = 0; y < Config::DIMY; ++y)
    for (unsigned int x = 0; x < Config::DIMX; ++x)
      outPede.write(
          reinterpret_cast<char *>(&(pedePtr[y * Config::DIMX + x].m2)),
          sizeof(pedePtr[y * Config::DIMX + x].m2));
  outPede.flush();
  outPede.close();
    


    


  std::ofstream energy_file("cluster_energy.bin", std::ios_base::binary);

  if (!energy_file.is_open()) {
    std::cerr << "Couldn't open energy output file!\n";
    abort();
  }
    
    int frameCounter = 0;
    
    // process data maps
    while (offset < data.numFrames) {

        // create views of the output arrays
        auto energy_view([&]() -> tl::optional<EnergyPackageView> {
            if (energy)
                return energy->getView(offset, Config::DEV_FRAMES);
            return tl::nullopt;
        }());

        auto photon_view([&]() -> tl::optional<PhotonPackageView> {
            if (photon)
                return photon->getView(offset, Config::DEV_FRAMES);
            return tl::nullopt;
        }());

        auto sum_view([&]() -> tl::optional<SumPackageView> {
            if (sum)
                return sum->getView(
                    (offset + Config::SUM_FRAMES - 1) / Config::SUM_FRAMES,
                    (Config::DEV_FRAMES + Config::SUM_FRAMES - 1) /
                        Config::SUM_FRAMES);
            return tl::nullopt;
        }());

        auto maxValues_view([&]() -> tl::optional<MaxValuePackageView> {
            if (maxValues)
                return maxValues->getView(offset, Config::DEV_FRAMES);
            return tl::nullopt;
        }());

        // save the upload future
        futures.emplace_back(dispenser.process(data,
                                               offset,
                                               ef,
                                               energy_view,
                                               photon_view,
                                               sum_view,
                                               maxValues_view,
                                               clusters));

        // get the number of frames that have been uploaded and add them to the
        // offset
        offset = std::get<0>(*futures.rbegin());



        std::get<1>(futures.back()).wait();
        dispenser.synchronize();
        
        save_image<Config>("pedestal_" + std::to_string(frameCounter),
            alpakaNativePtr(dispenser.downloadPedestaldata().data), 0);

        if (energy)
          save_image<Config>("energy_" + std::to_string(frameCounter),
                             alpakaNativePtr(energy_view->data), 0);

        
      // save energy data if available                                                                                                                                                                                                                       
      if (energy)
        energy_file.write(
            reinterpret_cast<char *>(alpakaNativePtr(energy->data)),
            sizeof(typename Config::EnergyMap) * offset);

        ++frameCounter;
        
        // print status message
        DEBUG(offset, "/", data.numFrames, "enqueued");
    }

    // wait for calculation to finish
    std::get<1>(futures.back()).wait();

    
    // save clusters (currently only used for debugging)
    if (clusters) {
        saveClustersBin<Config, ConcreteAcc>("clusters.bin", *clusters);
    }
    
    /*
    // save second last images
    if (energy)
        save_image<Config>(
            "eend2", alpakaNativePtr(energy->data), energy->numFrames - 2);

    if (photon)
        save_image<Config>(
            "pend2", alpakaNativePtr(photon->data), photon->numFrames - 2);

    // save last images
    if (energy)
        save_image<Config>(
            "eend", alpakaNativePtr(energy->data), energy->numFrames - 1);

    if (photon)
        save_image<Config>(
            "pend", alpakaNativePtr(photon->data), photon->numFrames - 1);

    if (sum)
        save_image<Config>(
            "send", alpakaNativePtr(sum->data), sum->numFrames - 1);

    if (maxValues)
        DEBUG("max value:",
              alpakaNativePtr(maxValues->data)[maxValues->numFrames - 1]);

    save_image<Config>(
        "pedestal", alpakaNativePtr(dispenser.downloadPedestaldata().data), 0);
    save_image<Config>(
        "gainstage", alpakaNativePtr(dispenser.downloadGainStages().data), 0);
    save_single_map<Config>("drift", dispenser.downloadDriftMap()->data);
    save_single_map<Config>("mask", dispenser.downloadMask()->data);

    // save clusters (currently only used for debugging)
    if (clusters) {
        saveClusters<Config, ConcreteAcc>("clusters.txt", *clusters);
        saveClustersBin<Config, ConcreteAcc>("clusters.bin", *clusters);
    }

*/

    // save clusters (currently only used for debugging)
    if (clusters) {
        saveClusters<Config, ConcreteAcc>("clusters.txt", *clusters);
        saveClustersBin<Config, ConcreteAcc>("clusters.bin", *clusters);
    }
    
    // print out GPU memory usage
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
