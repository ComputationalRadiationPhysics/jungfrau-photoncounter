#include "Filecache.hpp"
#include "Upload.hpp"
#include <iomanip>
#include <iostream>
#include <string>

int main()
{
    DEBUG("Entering main ...");
    Filecache fc(1024UL * 1024 * 1024 * 16);
    DEBUG("fc allocated. Loading maps!");
    // TODO: load pedestal init files and calibrate pedestal maps
    // std::vector<Datamap>
    // fc.loadMaps("data_pool/px_101016/allpede_250us_1243__B_000000.dat", 1024,
    // 512);
    std::vector<Pedestalmap> pedestal =
        fc.loadMaps<Pedestalmap>("data_pool/px_101016/pedeMaps.bin", 1024, 512);
    DEBUG("Pedestalmap loaded!");
    std::vector<Datamap> data = fc.loadMaps<Datamap>(
        "data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat", 1024, 512);
    DEBUG("Datamap loaded!");
    std::vector<Gainmap> gain = fc.loadMaps<Gainmap>(
        "data_pool/px_101016/gainMaps_M022.bin", 1024, 512);
    DEBUG("Gainmap loaded!");

    std::array<Pedestalmap, 3> pedestal_array = {pedestal[0], pedestal[1],
                                                 pedestal[2]};
    DEBUG("Pedestal array created!");

    std::array<Gainmap, 3> gain_array = {gain[0], gain[1], gain[2]};
    DEBUG("Gain array created!");

    Uploader up(gain_array, pedestal_array, 1024, 512);
    DEBUG("Uploader created!");

    up.upload(data);
    DEBUG("Uploaded 1/5!");
    up.upload(data);
    DEBUG("Uploaded 2/5!");
    up.upload(data);
    DEBUG("Uploaded 3/5!");
    up.upload(data);
    DEBUG("Uploaded 4/5!");
    up.upload(data);
    DEBUG("Uploaded 5/5!");

    return 0;
}
