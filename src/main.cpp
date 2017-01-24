#include "Bitmap.hpp"

int main()
{
	Bitmap::Image img(1024, 512);
	Bitmap::Rgb black = { 0, 0, 0 };
	Bitmap::Rgb white = { 255, 255, 255 };
	for (int x = 0; x < 1024; ++x) {
		for (int y = 0; y < 512; ++y) {
			img(x, y) = black;
		}
	}
	img.writeToFile("/home/flow/Desktop/test.bmp");
	return 0;
}

