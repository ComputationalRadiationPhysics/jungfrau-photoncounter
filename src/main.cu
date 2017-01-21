#include "Filecache.hpp"
#include "Upload.hpp"
#include <iomanip>
#include <iostream>
#include <string>

const std::size_t NUM_UPLOADS = 5;

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

	int num = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&num));

	DEBUG("# of CUDA devices: " << num);
    Uploader up(gain_array, pedestal_array, 1024, 512, num);
    DEBUG("Uploader created!");

	std::vector<Datamap> data_backup(data);
	std::vector<Photonmap> ready;
	ready.reserve(GPU_FRAMES);

	/*    DEBUG("Upload 1/5!");
	data = data_backup;
    DEBUG((up.upload(data) ? "done" : "failed"));*/
	DEBUG("starting upload!");
	for(std::size_t i = 0; i < NUM_UPLOADS; ++i) {
		while(!up.upload(data)) {
			while(!(ready = up.download()).empty()) {
				free(ready[0].data());
				DEBUG("freeing in main");
			}
			DEBUG("uploading again ...");
		}
		data = data_backup;
		DEBUG("Uploaded " << i << "/" << NUM_UPLOADS);
	}
/*
    DEBUG("Upload 2/5!");
	data = data_backup;
    DEBUG((up.upload(data) ? "done" : "failed"));
    DEBUG("Upload 3/5!");
	data = data_backup;
    DEBUG((up.upload(data) ? "done" : "failed"));
    DEBUG("Upload 4/5!");
	data = data_backup;
    DEBUG((up.upload(data) ? "done" : "failed"));
    DEBUG("Upload 5/5!");
	data = data_backup;
    DEBUG((up.upload(data) ? "done" : "failed"));*/

	up.synchronize();

    return 0;
}
