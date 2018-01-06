#include "Alpakaconfig.hpp"
#include "Config.hpp"
#include "Dispenser.hpp"
#include "Filecache.hpp"

//#include <thread>

using Accelerator = GpuCudaRt;

    /*    Accelerator alpkStr;
        std::shared_ptr<Dispenser<Accelerator>> dispenserPtr =
            std::make_shared<Dispenser<Accelerator>>(gain, alpkStr);

        dispenserPtr->uploadPedestaldata(pedestaldata);

        std::thread t1(upload, dispenserPtr, data);
        std::thread t2(download, dispenserPtr, data.numMaps);
        dispenserPtr.reset();

        t1.join();
        t2.join();
    */
/*auto upload(std::shared_ptr<Dispenser<Accelerator>> disp, Maps<Data> data)
    -> void
{
    std::shared_ptr<Dispenser<Accelerator>> dispenser = disp;
    std::size_t offset = 0;
    while (offset < data.numMaps) {
        offset = dispenser->uploadData(data, offset);
    }
    return;
}

auto download(std::shared_ptr<Dispenser<Accelerator>> disp, std::size_t numMaps)
    -> void
{
    std::shared_ptr<Dispenser<Accelerator>> dispenser = disp;
    Maps<Photon> photon{};
    Maps<PhotonSum> sum{};
    std::size_t downloaded = 0;
    while (downloaded < numMaps) {
        if (dispenser->downloadData(&photon, &sum)) {
            downloaded += DEV_FRAMES;
            DEBUG(downloaded << "/" << numMaps << " downloaded");
        }
    }
    return;
}*/

auto main() -> int
{
    Filecache fc(1024UL * 1024 * 1024 * 16);
    DEBUG("filecache created");

    Maps<Data, Accelerator> pedestaldata(
        fc.loadMaps<Data, Accelerator>("../../jungfrau-photoncounter/data_pool/px_101016/"
                          "allpede_250us_1243__B_000000.dat",
                          true));
    DEBUG(pedestaldata.numMaps << " pedestaldata maps loaded");

    Maps<Data, Accelerator> data(
        fc.loadMaps<Data, Accelerator>("../../jungfrau-photoncounter/data_pool/px_101016/"
                          "Insu_6_tr_1_45d_250us__B_000000.dat",
                          true));
    DEBUG(data.numMaps << " data maps loaded");

    Maps<Gain, Accelerator> gain(fc.loadMaps<Gain, Accelerator>(
        "../../jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"));
    DEBUG(gain.numMaps << " gain maps loaded");

    DEBUG("gpu count: "
          << (alpaka::pltf::getDevCount<alpaka::pltf::Pltf<alpaka::dev::Dev<
                  alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>,
                                            std::size_t>>>>()));

    Accelerator alpkStr;
    Dispenser<Accelerator>* dispenser = new Dispenser<Accelerator>(gain, alpkStr);

    dispenser->uploadPedestaldata(pedestaldata);

    Maps<Photon, Accelerator> photon{};
    Maps<PhotonSum, Accelerator> sum{};
    std::size_t offset = 0;
    std::size_t downloaded = 0;
    while (downloaded < data.numMaps) {
        offset = dispenser->uploadData(data, offset);
        if (dispenser->downloadData(&photon, &sum)) {
            downloaded += DEV_FRAMES;
            DEBUG(downloaded << "/" << data.numMaps << " downloaded");
        }
}
DEBUG("möp");
    return 0;
}
