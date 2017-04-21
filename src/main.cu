#include "Filecache.hpp"
#include "Upload.hpp"
#include "Bitmap.hpp"
#include <iomanip>
#include <iostream>
#include <string>

const std::size_t NUM_UPLOADS = 2;

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

	Bitmap::Image img(1024, 512);
	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = data(j, k, 0) / 256;
			Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
			img(j, k) = color;
		}
	}
	img.writeToFile("dtest.bmp");


	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = data(j, k, 1) / 256;
			Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
			img(j, k) = color;
		}
	}
	img.writeToFile("dtest1.bmp");

	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = data(j, k, 2) / 256;
			Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
			img(j, k) = color;
		}
	}
	img.writeToFile("dtest2.bmp");

	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = data(j, k, 3) / 256;
			Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
			img(j, k) = color;
		}
	}
	img.writeToFile("dtest3.bmp");

	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = data(j, k, 999) / 256;
			Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
			img(j, k) = color;
		}
	}
	img.writeToFile("dtest999.bmp");

	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = data(j, k, 1000) / 256;
			Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
			img(j, k) = color;
		}
	}
	img.writeToFile("dtest1000.bmp");


	Bitmap::Image img2(1024, 512);
	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = pedestal(j, k, 0) / 256;
			Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
			img2(j, k) = color;
		}
	}
	img2.writeToFile("ptest.bmp");

	Bitmap::Image img3(1024, 512);
	for(int j = 0; j < 1024; j++) {
		for(int k=0; k < 512; k++) {
			int h = gain(j, k, 0) * 200;
			Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
			img3(j, k) = color;
		}
	}
	img3.writeToFile("gtest.bmp");


	int num = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&num));

	//for debug only:
	num = 2;

    Uploader up(gain, pedestal, num);
    DEBUG("Uploader created!");
	Photonmap ready(0, NULL);

	DEBUG("starting upload!");

    int is_first_package = 1;
	for(std::size_t i = 1; i <= NUM_UPLOADS; ++i) {
		size_t current_offset = 0;
		while((current_offset = up.upload(data, current_offset)) < data.getSizeBytes() / DIMX / DIMY) {
			while(!(ready = up.download()).getSizeBytes()) {

                if (is_first_package == 1) {
                    Bitmap::Image img(1024, 512);
                    for(int j = 0; j < 1024; j++) {
                        for(int k=0; k < 512; k++) {
                            int h = ready(j, k, 0) / 256;
                            Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
                            img(j, k) = color;
                        }
                    }
                    img.writeToFile("test.bmp");


                    for(int j = 0; j < 1024; j++) {
                        for(int k=0; k < 512; k++) {
                            int h = ready(j, k, 2) / 256;
                            Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
                            img(j, k) = color;
                        }
                    }
                    img.writeToFile("test1.bmp");

                    for(int j = 0; j < 1024; j++) {
                        for(int k=0; k < 512; k++) {
                            int h = ready(j, k, 2) / 256;
                            Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
                            img(j, k) = color;
                        }
                    }
                    img.writeToFile("test2.bmp");

                    for(int j = 0; j < 1024; j++) {
                        for(int k=0; k < 512; k++) {
                            int h = ready(j, k, 3) / 256;
                            Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
                            img(j, k) = color;
                        }
                    }
                    img.writeToFile("test3.bmp");

                    for(int j = 0; j < 1024; j++) {
                        for(int k=0; k < 512; k++) {
                            int h = ready(j, k, 999) / 256;
                            Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
                            img(j, k) = color;
                        }
                    }
                    img.writeToFile("test999.bmp");



                    is_first_package = 2;
                } else if (is_first_package == 2){
                    Bitmap::Image img(1024, 512);
                    for(int j = 0; j < 1024; j++) {
                        for(int k=0; k < 512; k++) {
                            int h = ready(j, k, 0) / 256;
                            Bitmap::Rgb color = {(unsigned char)h, (unsigned char)h, (unsigned char)h};
                            img(j, k) = color;
                        }
                    }
                    img.writeToFile("test1000.bmp");

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
