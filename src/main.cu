#include "Filecache.hpp"
#include "Upload.hpp"
#include <iomanip>
#include <iostream>
#include <string>

const std::size_t NUM_UPLOADS = 10;

int main()
{
    DEBUG("Entering main ...");
    Filecache fc(1024UL * 1024 * 1024 * 16);
    DEBUG("fc allocated. Loading maps!");
    std::vector<Pedestalmap> pedestal =
        fc.loadMaps<Pedestalmap>("data_pool/px_101016/pedeMaps.bin", 1024, 512);
    DEBUG("Pedestalmap loaded!");
    std::vector<Datamap> data = fc.loadMaps<Datamap>("data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat", 1024, 512);
    DEBUG("Datamap [NOT] loaded!");
    std::vector<Gainmap> gain = fc.loadMaps<Gainmap>(
        "data_pool/px_101016/gainMaps_M022.bin", 1024, 512);
    DEBUG("Gainmap loaded!");

    std::array<Pedestalmap, 3> pedestal_array = {pedestal[0], pedestal[1],
                                                 pedestal[2]};
    DEBUG("Pedestal array created!");

    std::array<Gainmap, 3> gain_array = {gain[0], gain[1], gain[2]};
    DEBUG("Gain array created!");

	int num = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&num));

	DEBUG("# of CUDA devices: " << num);
    Uploader up(gain_array, pedestal_array, 1024, 512, num);
    DEBUG("Uploader created!");

	std::vector<Datamap> data_backup(data);
	std::vector<Photonmap> ready;
	ready.reserve(GPU_FRAMES);

	std::size_t remove_me = 0;
	DEBUG("starting upload!");
	for(std::size_t i = 1; i <= NUM_UPLOADS; ++i) {
		while(!up.upload(data) && !data.empty()) {
			while(!(ready = up.download()).empty()) {
				free(ready[0].data());
				DEBUG("freeing in main");
			}
			DEBUG("data size: " << data.size());
			//DEBUG("data_backup size: " << data_backup.size());
			DEBUG("uploading again (" << remove_me++ << ") ...");
		}
		data = data_backup;
		DEBUG("Uploaded " << i << "/" << NUM_UPLOADS);
		remove_me = 0;
	}

	up.synchronize();

    return 0;
}
