#include "Bitmap.hpp"
#include "Filecache.hpp"
#include "cuda/Upload.hpp"
#include <iomanip>
#include <iostream>
#include <string>

const std::size_t NUM_UPLOADS = 1;

template<typename Maptype> void save_image(std::string path, Maptype map, std::size_t frame_number, double divider = 128) {
	Bitmap::Image img(1024, 512);
	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = map(j, k, frame_number) / divider;
             Bitmap::Rgb color = {(unsigned char)(h & 255), 
                (unsigned char)((h >> 8) & 255),
                (unsigned char)((h >> 16) & 255)};
			img(j, k) = color;
		}
	}
	img.writeToFile(path);
}

int main()
{
    DEBUG("Entering main ...");
    Filecache fc(1024UL * 1024 * 1024 * 16);
    Datamap pedestaldata = fc.loadMaps<Datamap>(
        "data_pool/px_101016/allpede_250us_1243__B_000000.dat", true);
    Datamap data = fc.loadMaps<Datamap>(
        "data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat", true);
    Gainmap gain =
        fc.loadMaps<Gainmap>("data_pool/px_101016/gainMaps_M022.bin");
    DEBUG("Maps loaded!");

    save_image<Datamap>(std::string("test21.bmp"), data, std::size_t(21));

    int num = 0;
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&num));

    Uploader up(gain, num);
    DEBUG("Uploader created!");
    up.uploadPedestaldata(pedestaldata);
    DEBUG("Pedestals created!");
    Photonmap ready(0, NULL, true);

	DEBUG("starting uploading " << data.getN() << " maps!");
	
    int is_first_package = 1;
	for(std::size_t i = 1; i <= NUM_UPLOADS; ++i) {
		size_t current_offset = 0;
		while((current_offset = up.upload(data, current_offset)) < data.getN()) {

			std::size_t internal_offset = 0;
			
			while(!up.isEmpty()) {
				if(!(ready = up.download()).getN())
					continue;
 
				internal_offset += ready.getN();
				
				DEBUG(internal_offset << " of " << current_offset << " Frames processed!");
				
                if (is_first_package == 1) {

                    save_image<Datamap>(std::string("dtest.bmp"), ready,
                                        std::size_t(0));
                    save_image<Datamap>(std::string("dtest1.bmp"), ready,
                                        std::size_t(1));
                    save_image<Datamap>(std::string("dtest2.bmp"), ready,
                                        std::size_t(2));
                    save_image<Datamap>(std::string("dtest3.bmp"), ready,
                                        std::size_t(3));
                    save_image<Datamap>(std::string("dtest500.bmp"), ready,
                                        std::size_t(500));
                    save_image<Datamap>(std::string("dtest999.bmp"), ready,
                                        std::size_t(999));

                    is_first_package = 2;
                } else if (is_first_package == 2){
					
					save_image<Datamap>(std::string("dtest1000.bmp"), ready, std::size_t(0));
					save_image<Datamap>(std::string("dtest1001.bmp"), ready, std::size_t(1));
					save_image<Datamap>(std::string("dtest1002.bmp"), ready, std::size_t(2));
					save_image<Datamap>(std::string("dtest1003.bmp"), ready, std::size_t(3));
					save_image<Datamap>(std::string("dtest1004.bmp"), ready, std::size_t(4));
					save_image<Datamap>(std::string("dtest1005.bmp"), ready, std::size_t(5));
					save_image<Datamap>(std::string("dtest1006.bmp"), ready, std::size_t(6));
				    is_first_package = 0;
				}
				free(ready.data());
				DEBUG("freeing in main");
			}
		}
		DEBUG("Uploaded " << i << "/" << NUM_UPLOADS);
	}

	DEBUG("Flushing leftover frames");
	
	up.synchronize();

	std::size_t internal_offset = 0;	
	while(!up.isEmpty()) {
		if(!(ready = up.download()).getN())
			continue;

		internal_offset += ready.getN();
				
		DEBUG(internal_offset << " Frames processed!");

		free(ready.data());
		DEBUG("freeing leftover frames");
	}

    return 0;
}
