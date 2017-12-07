#include "Config.hpp"
#include "Alpakaconfig.hpp"
#include "Filecache.hpp"
#include "Dispenser.hpp"


auto main() -> int
{
    Filecache fc(1024UL * 1024 * 1024 * 5);
    DEBUG("filecache created");

    Maps<Data> pedestaldata(fc.loadMaps<Data>(
        "../data_pool/px_101016/allpede_250us_1243__B_000000.dat", true));
    DEBUG(pedestaldata.numMaps << " pedestaldata maps loaded");

    /*Maps<Data> data(fc.loadMaps<Data>(
        "../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat", true));
    DEBUG(data.numMaps << " data maps loaded");*/

    Maps<Data> data(
        fc.loadMaps<Data>("../data_pool/px_101016/1kFrames.dat", true));
    DEBUG(data.numMaps << " data maps loaded");

    Maps<Gain> gain(
        fc.loadMaps<Gain>("../data_pool/px_101016/gainMaps_M022.bin"));
    DEBUG(gain.numMaps << " gain maps loaded");

    CpuSerial alpkStr;
    Dispenser<CpuSerial> dispenser(&gain, alpkStr);
    
    dispenser.uploadPedestaldata(&pedestaldata);
    dispenser.uploadData(&data);

    Maps<Photon>* photon{};
    Maps<PhotonSum>* sum{};
    std::size_t downloaded = 0;
    while(downloaded < data.numMaps) {
        DEBUG("downloading...");
        if(dispenser.downloadData(photon, sum)){
            downloaded += DEV_FRAMES; 
        }
    }

    save_image<Photon>(
        static_cast<std::string>("TestframeLast.bmp"),
        photon->dataPointer,
        DEV_FRAMES);

    return 0;
}
