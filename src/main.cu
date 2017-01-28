#include "Filecache.hpp"
#include "Upload.hpp"
#include "Bitmap.hpp"
#include <iomanip>
#include <iostream>
#include <string>

const std::size_t NUM_UPLOADS = 10;

int main()
{
    DEBUG("Entering main ...");
    Filecache fc(1024UL * 1024 * 1024 * 16);
    std::vector<Pedestalmap> pedestal =
        fc.loadMaps<Pedestalmap>("data_pool/px_101016/pedeMaps.bin", 1024, 512);
    DEBUG("Pedestalmap loaded!");
    std::vector<Datamap> data = fc.loadMaps<Datamap>("data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat", 1024, 512);
    DEBUG("Datamap loaded!");
    std::vector<Gainmap> gain = fc.loadMaps<Gainmap>(
        "data_pool/px_101016/gainMaps_M022.bin", 1024, 512);
    DEBUG("Gainmap loaded!");

    std::array<Pedestalmap, 3> pedestal_array = {pedestal[0], pedestal[1], pedestal[2]};
    std::array<Gainmap, 3> gain_array = {gain[0], gain[1], gain[2]};

	int num = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&num));

    Uploader up(gain_array, pedestal_array, 1024, 512, num);
    DEBUG("Uploader created!");

	std::vector<Datamap> data_backup(data);
	std::vector<Photonmap> ready;
	ready.reserve(GPU_FRAMES);

	DEBUG("starting upload!");

    int bitteFunktioniere = 1;
	for(std::size_t i = 1; i <= NUM_UPLOADS; ++i) {
		while(!up.upload(data) && !data.empty()) {
			while(!(ready = up.download()).empty()) {
                Photonmap test = ready.at(0);
                if (bitteFunktioniere == 1) {
                    Bitmap::Image img(1024, 512);
                    for(int j = 0; j < 1024; j++) {
                        for(int k=0; k < 512; k++) {
                            int h = test(j, k) / 256;
                            Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
                            img(j, k) = color;
                        }
                    }
                    img.writeToFile("test.bmp");
                    bitteFunktioniere = 0;
                }
				free(ready[0].data());
				DEBUG("freeing in main");
			}
		}
		data = data_backup;
		DEBUG("Uploaded " << i << "/" << NUM_UPLOADS);
	}

	up.synchronize();

    return 0;
}
