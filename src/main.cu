#include "Bitmap.hpp"
#include "Filecache.hpp"
#include "Upload.hpp"
#include <iomanip>
#include <iostream>
#include <string>

const std::size_t NUM_UPLOADS = 2;

template<typename Maptype> void save_image(std::string path, Maptype map, std::size_t frame_number, double divider = 256) {
	Bitmap::Image img(1024, 512);
	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = map(j, k, frame_number) / divider;
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
    Datamap pedestaldata = fc.loadMaps<Datamap>(
        "data_pool/px_101016/allpede_250us_1243__B_000000.dat");
    Datamap data = fc.loadMaps<Datamap>(
        "data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat", true);
    Gainmap gain =
        fc.loadMaps<Gainmap>("data_pool/px_101016/gainMaps_M022.bin");
    DEBUG("Maps loaded!");

    save_image<Datamap>(std::string("test.bmp"), data, std::size_t(0));
    save_image<Datamap>(std::string("test1.bmp"), data, std::size_t(1));
    save_image<Datamap>(std::string("test2.bmp"), data, std::size_t(2));
    save_image<Datamap>(std::string("test3.bmp"), data, std::size_t(3));
    save_image<Datamap>(std::string("test4.bmp"), data, std::size_t(4));
    save_image<Datamap>(std::string("test5.bmp"), data, std::size_t(5));
    save_image<Datamap>(std::string("test6.bmp"), data, std::size_t(6));
    save_image<Datamap>(std::string("test7.bmp"), data, std::size_t(7));
    save_image<Datamap>(std::string("test8.bmp"), data, std::size_t(8));
    save_image<Datamap>(std::string("test9.bmp"), data, std::size_t(9));
    save_image<Datamap>(std::string("test10.bmp"), data, std::size_t(10));
    save_image<Datamap>(std::string("test11.bmp"), data, std::size_t(11));
    save_image<Datamap>(std::string("test12.bmp"), data, std::size_t(12));
    save_image<Datamap>(std::string("test13.bmp"), data, std::size_t(13));
    save_image<Datamap>(std::string("test14.bmp"), data, std::size_t(14));
    save_image<Datamap>(std::string("test15.bmp"), data, std::size_t(15));
    save_image<Datamap>(std::string("test16.bmp"), data, std::size_t(16));
    save_image<Datamap>(std::string("test17.bmp"), data, std::size_t(17));
    save_image<Datamap>(std::string("test18.bmp"), data, std::size_t(18));
    save_image<Datamap>(std::string("test19.bmp"), data, std::size_t(19));
    save_image<Datamap>(std::string("test20.bmp"), data, std::size_t(20));
    save_image<Datamap>(std::string("test21.bmp"), data, std::size_t(21));
    save_image<Datamap>(std::string("test22.bmp"), data, std::size_t(22));
    save_image<Datamap>(std::string("test23.bmp"), data, std::size_t(23));
    save_image<Datamap>(std::string("test24.bmp"), data, std::size_t(24));
    save_image<Datamap>(std::string("test500.bmp"), data, std::size_t(500));
    save_image<Datamap>(std::string("test999.bmp"), data, std::size_t(999));
    save_image<Datamap>(std::string("test1000.bmp"), data, std::size_t(1000));

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

	up.downloadPedestalmap();
	
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
