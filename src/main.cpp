#include "Alpakaconfig.hpp"
#include "Config.hpp"
#include "Dispenser.hpp"
#include "Filecache.hpp"

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */
using Accelerator = GpuCudaRt;

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

    FramePackage<MaskMap, Accelerator, Dim, Size> mask;/*(
        fc->loadMaps<MaskMap, Accelerator, Dim, Size>(
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

    alpaka::mem::buf::Buf<typename Accelerator::DevHost, MaskMap, Dim, Size>
        maskPtr(
            mask.numFrames == 1
                ? mask.data
                : alpaka::mem::buf::alloc<MaskMap, Size>(
                      alpaka::pltf::getDevByIdx<typename Accelerator::PltfHost>(
                          0u),
                      0lu));

    Dispenser<Accelerator, Dim, Size>* dispenser =
      new Dispenser<Accelerator, Dim, Size>(gain, maskPtr);


    // upload and calculate pedestal data
    dispenser->uploadPedestaldata(pedestaldata);

    FramePackage<PhotonMap, Accelerator, Dim, Size> photon{};
    FramePackage<PhotonSumMap, Accelerator, Dim, Size> sum{};
    std::size_t offset = 0;
    std::size_t downloaded = 0;

    // process data maps
    while (downloaded < data.numFrames) {
        offset = dispenser->uploadData(data, offset);
        if (dispenser->downloadData(&photon, &sum)) {
            downloaded += DEV_FRAMES;
            DEBUG(downloaded << "/" << data.numFrames << " downloaded");
        }
    }
    
    /*
    save_image<PedestalMap>(
        static_cast<std::string>(std::to_string(dev->id) + "pedestal" +
                                 std::to_string(std::rand() % 1000)),
        alpaka::mem::view::getPtrNative(dev->energyHost),
        DEV_FRAMES - 1);
    */

    auto sizes = dispenser->getMemSize();
    auto free_mem = dispenser->getFreeMem();

    for (std::size_t i = 0; i < sizes.size(); ++i)
        DEBUG("Device #" << i << ": "
                         << (float)free_mem[i] / (float)sizes[i] * 100.0f
                         << "% free of " << sizes[i] << " Bytes");

    return 0;
}
