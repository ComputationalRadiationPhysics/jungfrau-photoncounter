#include "Alpakaconfig.hpp"
#include "Config.hpp"
#include "Dispenser.hpp"
#include "Filecache.hpp"

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */
using Accelerator = GpuCudaRt;//CpuSerial;

auto main() -> int
{
    // t is used in all debug-messages
    t = Clock::now();

    Filecache* fc = new Filecache(1024UL * 1024 * 1024 * 16);
    DEBUG("filecache created");

    // load maps
    FramePackage<DetectorData, Accelerator, Dim, Size> pedestaldata(
        fc->loadMaps<DetectorData, Accelerator, Dim, Size>(
            "../../jungfrau-photoncounter/data_pool/px_101016/"
            "allpede_250us_1243__B_000000.dat",
            true));
    DEBUG(pedestaldata.numFrames << " pedestaldata maps loaded");

    FramePackage<DetectorData, Accelerator, Dim, Size> data(
        fc->loadMaps<DetectorData, Accelerator, Dim, Size>(
            "../../jungfrau-photoncounter/data_pool/px_101016/"
            "Insu_6_tr_1_45d_250us__B_000000.dat",
            true));
    DEBUG(data.numFrames << " data maps loaded");

    FramePackage<GainMap, Accelerator, Dim, Size> gain(fc->loadMaps<GainMap,
                                                                    Accelerator,
                                                                    Dim,
                                                                    Size>(
        "../../jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"));
    DEBUG(gain.numFrames << " gain maps loaded");

    FramePackage<MaskMap, Accelerator, Dim, Size> mask(SINGLEMAP);
    mask.numFrames = 0;
    /*(fc->loadMaps<MaskMap, Accelerator, Dim, Size>(
            "../data_pool/px_101016/mask.bin"));
            DEBUG(mask.numFrames << " masking maps loaded");*/
    delete (fc);

    // print info
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    DEBUG("gpu count: "
          << (alpaka::pltf::getDevCount<alpaka::pltf::Pltf<alpaka::dev::Dev<
                  alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>,
                                            std::size_t>>>>()));
#endif
    DEBUG("cpu count: " << (alpaka::pltf::getDevCount<
                            alpaka::pltf::Pltf<typename Accelerator::Acc>>()));

    boost::optional<alpaka::mem::buf::Buf<typename Accelerator::DevHost, MaskMap, Dim, Size>> maskPtr;
    if(mask.numFrames == SINGLEMAP)
      maskPtr = mask.data;

    //! @todo: throw this new out
    Dispenser<Accelerator, Dim, Size>* dispenser =
      new Dispenser<Accelerator, Dim, Size>(gain, maskPtr);


    // upload and calculate pedestal data
    dispenser->uploadPedestaldata(pedestaldata);

    FramePackage<PhotonMap, Accelerator, Dim, Size> photon(DEV_FRAMES);
    FramePackage<PhotonSumMap, Accelerator, Dim, Size> sum(DEV_FRAMES / SUM_FRAMES);
    ClusterArray<Accelerator, Dim, Size> clusters;
    FramePackage<EnergyValue, Accelerator, Dim, Size> maxValues(DEV_FRAMES);
    std::size_t offset = 0;
    std::size_t downloaded = 0;

    // process data maps
    while (downloaded < data.numFrames) {
        offset = dispenser->uploadData(data, offset);
        if (dispenser->downloadData(photon, sum, maxValues, clusters)) {
          //! @todo: only correct if DEV_FRAMES were actually uploaded. 
            downloaded += DEV_FRAMES;
            DEBUG(downloaded << "/" << data.numFrames << " downloaded");
        }
    }
    
    saveClusters("clusters.bin", clusters);

    GainStageMap* gainStage = dispenser->downloadGainStages();
    save_image<GainStageMap>("gainstage", gainStage, 0);

    DriftMap* drift = dispenser->downloadDriftMaps();
    save_image<DriftMap>("driftmap", drift, 0);

    for(uint32_t i = 0; i < maxValues.numFrames; ++i)
      DEBUG("max value for frame " << i << ": " << alpaka::mem::view::getPtrNative(maxValues.data)[i]);

    
    auto sizes = dispenser->getMemSize();
    auto free_mem = dispenser->getFreeMem();

    for (std::size_t i = 0; i < sizes.size(); ++i)
        DEBUG("Device #" << i << ": "
                         << (float)free_mem[i] / (float)sizes[i] * 100.0f
                         << "% free of " << sizes[i] << " Bytes");

    return 0;
}
