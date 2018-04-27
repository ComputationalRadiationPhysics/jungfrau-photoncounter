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
    //t is used in all debug-messages
    t = Clock::now();

    Filecache* fc = new Filecache(1024UL * 1024 * 1024 * 16);
    DEBUG("filecache created");

	//load maps
    Maps<Data, Accelerator> pedestaldata(
        fc->loadMaps<Data, Accelerator>("../../jungfrau-photoncounter/data_pool/px_101016/"
                          "allpede_250us_1243__B_000000.dat",
                          true));
    DEBUG(pedestaldata.numMaps << " pedestaldata maps loaded");

    Maps<Data, Accelerator> data(
        fc->loadMaps<Data, Accelerator>("../../jungfrau-photoncounter/data_pool/px_101016/"
                          "Insu_6_tr_1_45d_250us__B_000000.dat",
                          true));
    DEBUG(data.numMaps << " data maps loaded");

    Maps<Gain, Accelerator> gain(fc->loadMaps<Gain, Accelerator>(
        "../../jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"));
    DEBUG(gain.numMaps << " gain maps loaded");
    delete(fc);

	//print info
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    DEBUG("gpu count: "
          << (alpaka::pltf::getDevCount<alpaka::pltf::Pltf<alpaka::dev::Dev<
                  alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>,
                                            std::size_t>>>>()));
#endif
    DEBUG("cpu count: "
          << (alpaka::pltf::getDevCount<alpaka::pltf::Pltf<typename Accelerator::Acc>>()));
	
    Dispenser<Accelerator>* dispenser = new Dispenser<Accelerator>(gain);
	
	DEBUG("Devices size: " << dispenser->getMemSize());
	DEBUG("Free sapce: " << dispenser->getFreeMem());

	//upload and calculate pedestal data
    dispenser->uploadPedestaldata(pedestaldata);

    Maps<Photon, Accelerator> photon{};
    Maps<PhotonSum, Accelerator> sum{};
    std::size_t offset = 0;
    std::size_t downloaded = 0;

	//process data maps
    while (downloaded < data.numMaps) {
        offset = dispenser->uploadData(data, offset);
        if (dispenser->downloadData(&photon, &sum)) {
            downloaded += DEV_FRAMES;
            DEBUG(downloaded << "/" << data.numMaps << " downloaded");
        }
	}

    return 0;
}
