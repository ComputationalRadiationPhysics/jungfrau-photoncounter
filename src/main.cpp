#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"
#include "jungfrau-photoncounter/Dispenser.hpp"

#include "Filecache.hpp"

#include "jungfrau-photoncounter/Debug.hpp"

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */
using Accelerator = GpuCudaRt; // CpuSerial;

auto main(int argc, char* argv[]) -> int
{
    // t is used in all debug-messages
    t = Clock::now();

    // create a file cache for all input files
    Filecache* fc = new Filecache(1024UL * 1024 * 1024 * 16);
    DEBUG("filecache created");

    // load maps
    FramePackage<DetectorData, Accelerator> pedestaldata(
        fc->loadMaps<DetectorData, Accelerator>(
            //"../../data_pool/px_101016/allpede_250us_1243__B_000000.dat",
            "../../../moench_data/"
            "1000_frames_pede_e17050_1_00018_00000.dat",
            true));
    DEBUG(pedestaldata.numFrames, "pedestaldata maps loaded");

    FramePackage<DetectorData, Accelerator> data(
        fc->loadMaps<DetectorData, Accelerator>(
            //"../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat",
            "../../../moench_data/"
            "e17050_1_00018_00000_image.dat",
            true));
    DEBUG(data.numFrames, "data maps loaded");

    FramePackage<GainMap, Accelerator> gain(
        fc->loadMaps<GainMap, Accelerator>(
            "../../../moench_data/moench_gain.bin"
            //"../../data_pool/px_101016/gainMaps_M022.bin"
            ));
    DEBUG(gain.numFrames, "gain maps loaded");

    FramePackage<MaskMap, Accelerator> mask(SINGLEMAP);
    mask.numFrames = 0;
    delete (fc);

    // create empty, optional input mask
    boost::optional<Accelerator::HostBuf<MaskMap>>
        maskPtr;
    if (mask.numFrames == SINGLEMAP)
        maskPtr = mask.data;

    Dispenser<Accelerator> dispenser(gain, maskPtr);

    // print info
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    DEBUG(
        "gpu count:",
        dispenser.getMemSize().size());
#endif
    
    // upload and calculate pedestal data
    dispenser.uploadPedestaldata(pedestaldata);

    // allocate space for output data
    FramePackage<EnergyMap, Accelerator> energy_data(DEV_FRAMES);
    FramePackage<PhotonMap, Accelerator> photon_data(DEV_FRAMES);
    FramePackage<SumMap, Accelerator> sum_data(DEV_FRAMES /
                                                          SUM_FRAMES);
    ClusterArray<Accelerator> clusters_data(30000 * 40000 / 50);
    FramePackage<EnergyValue, Accelerator> maxValues_data(
        DEV_FRAMES);

    boost::optional<FramePackage<EnergyMap, Accelerator>&> energy =
        energy_data;
    boost::optional<FramePackage<PhotonMap, Accelerator>&>
        photon; // = photon_data;
    boost::optional<FramePackage<SumMap, Accelerator>&> sum =
        sum_data;
    boost::optional<ClusterArray<Accelerator>&> clusters =
        clusters_data;

    boost::optional<FramePackage<EnergyValue, Accelerator>&>
        maxValues = maxValues_data;

    std::size_t offset = 0;
    std::size_t downloaded = 0;
    std::size_t currently_downloaded_frames = 0;

    PixelTracker<Accelerator> pt(argc, argv);

    ExecutionFlags ef;
    ef.mode = 2;
    ef.summation = 1;
    ef.masking = 1;
    ef.maxValue = 1;

    // process data maps
    while (downloaded < data.numFrames) {
        offset = dispenser.uploadData(data, offset, ef);
        if (currently_downloaded_frames = dispenser.downloadData(
                energy, photon, sum, maxValues, clusters)) {

            auto ipdata = dispenser.downloadInitialPedestaldata();
            auto pdata = dispenser.downloadPedestaldata();
            pt.push_back(ipdata, pdata, data, offset - 1);

            downloaded += currently_downloaded_frames;
            DEBUG(downloaded,
                  "/",
                  data.numFrames,
                  "downloaded;",
                  offset,
                  "uploaded");
        }
    }

    if (clusters) {
        saveClusters("clusters.txt", *clusters);
        saveClustersBin("clusters.bin", *clusters);
    }

    pt.save();

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
