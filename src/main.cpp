#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"
#include "jungfrau-photoncounter/Dispenser.hpp"

#include "Filecache.hpp"

#include "jungfrau-photoncounter/Debug.hpp"

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */
template <std::size_t MAPSIZE> using Accelerator = GpuCudaRt<MAPSIZE>;
using Config = JungfrauConfig;
using ConcreteAcc = Accelerator<Config::MAPSIZE>;

auto main(int argc, char* argv[]) -> int
{
    // t is used in all debug-messages
    t = Clock::now();

    // create a file cache for all input files
    Filecache<Config>* fc = new Filecache<Config>(1024UL * 1024 * 1024 * 16);
    DEBUG("filecache created");

    // load maps
    typename Config::FramePackage<typename Config::DetectorData, ConcreteAcc>
        pedestaldata(fc->loadMaps<typename Config::DetectorData, ConcreteAcc>(
            "../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat",
            //"../../../moench_data/"
            //"1000_frames_pede_e17050_1_00018_00000.dat",
            true));
    DEBUG(pedestaldata.numFrames, "pedestaldata maps loaded");

    typename Config::FramePackage<typename Config::DetectorData, ConcreteAcc>
        data(fc->loadMaps<typename Config::DetectorData, ConcreteAcc>(
            "../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat",
            //"../../../moench_data/"
            //"e17050_1_00018_00000_image.dat",
            true));
    DEBUG(data.numFrames, "data maps loaded");

    typename Config::FramePackage<typename Config::GainMap, ConcreteAcc> gain(
        fc->loadMaps<typename Config::GainMap, ConcreteAcc>(
            //"../../../moench_data/moench_gain.bin"
            "../../../data_pool/px_101016/gainMaps_M022.bin"));
    DEBUG(gain.numFrames, "gain maps loaded");

    typename Config::FramePackage<typename Config::MaskMap, ConcreteAcc> mask(
        Config::SINGLEMAP);
    mask.numFrames = 0;
    delete (fc);

    // create empty, optional input mask
    boost::optional<ConcreteAcc::HostBuf<typename Config::MaskMap>> maskPtr;
    if (mask.numFrames == Config::SINGLEMAP)
        maskPtr = mask.data;

    // initialize the dispenser
    Dispenser<Config, Accelerator> dispenser(gain, maskPtr);

    // print info
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    DEBUG("gpu count:", dispenser.getMemSize().size());
#endif

    // upload and calculate pedestal data
    dispenser.uploadPedestaldata(pedestaldata);

    // allocate space for output data
    typename Config::FramePackage<typename Config::EnergyMap, ConcreteAcc>
        energy_data(Config::DEV_FRAMES);
    typename Config::FramePackage<typename Config::PhotonMap, ConcreteAcc>
        photon_data(Config::DEV_FRAMES);
    typename Config::FramePackage<typename Config::SumMap, ConcreteAcc>
        sum_data(Config::DEV_FRAMES / Config::SUM_FRAMES);
    typename Config::ClusterArray<ConcreteAcc> clusters_data(30000 * 40000 /
                                                             50);
    typename Config::FramePackage<typename Config::EnergyValue, ConcreteAcc>
        maxValues_data(Config::DEV_FRAMES);

    boost::optional<
        typename Config::FramePackage<typename Config::EnergyMap, ConcreteAcc>&>
        energy = energy_data;
    boost::optional<
        typename Config::FramePackage<typename Config::PhotonMap, ConcreteAcc>&>
        photon; // = photon_data;
    boost::optional<
        typename Config::FramePackage<typename Config::SumMap, ConcreteAcc>&>
        sum = sum_data;
    typename Config::ClusterArray<ConcreteAcc>* clusters = &clusters_data;

    boost::optional<typename Config::FramePackage<typename Config::EnergyValue,
                                                  ConcreteAcc>&>
        maxValues = maxValues_data;

    // create variables to track the progress
    std::size_t offset = 0;
    std::size_t downloaded = 0;
    std::size_t currently_downloaded_frames = 0;

    // create vectors to hold the number of downloaded / uploaded frames and a
    // future The future is a variable, which is not "ready", when it is
    // returned from a function. Internally, this contains a set of instructions
    // to calculate the result. So if a future is accessed, the program waits
    // until the variable is processed. We use the future to signalize that an
    // upload or download operation is done. The internal value, which is then
    // returned, doesn't hold any deeper meaning. Once a future goes out pf
    // scope, it waits for its variable to be calculated. To delay this to the
    // end of the program, we create a vector of futures and store them there.
    std::vector<std::tuple<std::size_t, std::future<bool>>> uploadFutures;
    std::vector<std::tuple<std::size_t, std::future<bool>>> downloadFutures;

    // set execution flags
    typename Config::ExecutionFlags ef;
    ef.mode = 1;
    ef.summation = 0;
    ef.masking = 1;
    ef.maxValue = 0;

    // process data maps
    while (downloaded < data.numFrames) {
        // save the upload future
        uploadFutures.emplace_back(dispenser.uploadData(data, offset, ef));

        // get the number of frames that have been uploaded and add them to the
        // offset
        offset = std::get<0>(*uploadFutures.rbegin());

        // save the download future
        downloadFutures.emplace_back(
            dispenser.downloadData(energy, photon, sum, maxValues, clusters));

        // update the number of downloaded frames, if any frames where
        // downloaded
        currently_downloaded_frames = std::get<0>(*downloadFutures.rbegin());
        if (currently_downloaded_frames) {
            downloaded += currently_downloaded_frames;
            DEBUG(downloaded,
                  "/",
                  data.numFrames,
                  "downloaded;",
                  offset,
                  "uploaded");
        }
    }

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
