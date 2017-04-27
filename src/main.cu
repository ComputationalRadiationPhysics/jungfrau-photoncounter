#include "Filecache.hpp"
#include "Upload.hpp"
#include "Bitmap.hpp"
#include <iomanip>
#include <iostream>
#include <string>

const std::size_t NUM_UPLOADS = 2;

template<typename Maptype> void save_image(std::string path, Maptype map, std::size_t frame_number) {
	Bitmap::Image img(1024, 512);
	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = map(j, k, frame_number) / 256;
			Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
			img(j, k) = color;
		}
	}
	img.writeToFile(path);
}

int main()
{
    DEBUG("Entering main ...");
    Filecache fc(1024UL * 1024 * 1024 * 16);
    Pedestalmap pedestal_too_large = fc.loadMaps<Pedestalmap>("data_pool/px_101016/pedeMaps.bin");
    Datamap data = fc.loadMaps<Datamap>("data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat", true);
    Gainmap gain = fc.loadMaps<Gainmap>("data_pool/px_101016/gainMaps_M022.bin");
    DEBUG("Maps loaded!");

	Pedestalmap pedestal(3, pedestal_too_large.data(), false);


	//TODO: remove below; this is only used because the loaded pedestal maps seam to be incorrect
	//force pedestal to 0
	uint16_t* p = pedestal.data();
	for(std::size_t i = 0; i < pedestal.getSizeBytes(); ++i){
		p[i] = 0;
	}
    

	save_image<Datamap>(std::string("test.bmp"), data, std::size_t(0));
	save_image<Datamap>(std::string("test1.bmp"), data, std::size_t(1));
	save_image<Datamap>(std::string("test2.bmp"), data, std::size_t(2));
	save_image<Datamap>(std::string("test3.bmp"), data, std::size_t(3));
	save_image<Datamap>(std::string("test500.bmp"), data, std::size_t(500));
	save_image<Datamap>(std::string("test999.bmp"), data, std::size_t(999));
	save_image<Datamap>(std::string("test1000.bmp"), data, std::size_t(1000));
	save_image<Gainmap>(std::string("gtest.bmp"), gain, std::size_t(0));
	save_image<Pedestalmap>(std::string("ptest.bmp"), pedestal, std::size_t(0));

	int num = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&num));

    Uploader up(gain, pedestal, num);
    DEBUG("Uploader created!");
	Photonmap ready(0, NULL);	

	DEBUG("starting upload!");

    int is_first_package = 1;
	for(std::size_t i = 1; i <= NUM_UPLOADS; ++i) {
		size_t current_offset = 0;
		while((current_offset = up.upload(data, current_offset)) < 10000 - 1/*TODO: calculate number of frames*/) {

			std::size_t internal_offset = 0;
			
			while(!up.isEmpty()) {
				if(!(ready = up.download()).getSizeBytes())
					continue;

				internal_offset += ready.getSizeBytes() / DIMX / DIMY;
				
				DEBUG(internal_offset << " of " << current_offset << " Frames processed!");
				
                if (is_first_package == 1) {
					
					save_image<Datamap>(std::string("dtest.bmp"), ready, std::size_t(0));
					save_image<Datamap>(std::string("dtest1.bmp"), ready, std::size_t(1));
					save_image<Datamap>(std::string("dtest2.bmp"), ready, std::size_t(2));
					save_image<Datamap>(std::string("dtest3.bmp"), ready, std::size_t(3));
					save_image<Datamap>(std::string("dtest500.bmp"), ready, std::size_t(500));
					save_image<Datamap>(std::string("dtest999.bmp"), ready, std::size_t(999));

                    is_first_package = 2;
                } else if (is_first_package == 2){
					
					save_image<Datamap>(std::string("dtest1000.bmp"), data, std::size_t(0));
				    is_first_package = 0;
				}
				free(ready.data());
				DEBUG("freeing in main");
			}
		}
		DEBUG("Uploaded " << i << "/" << NUM_UPLOADS);
	}

	up.synchronize();

    return 0;
}
