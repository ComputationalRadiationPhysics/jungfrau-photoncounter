#include "Bitmap.hpp"

int main()
{
    Bitmap::Image img(1024, 512);
    Bitmap::Rgb black = {0, 0, 0};
    Bitmap::Rgb white = {255, 255, 255};
    for (int x = 0; x < 4; ++x) {
        for (int y = 0; y < 4; ++y) {
            img(x, y) = black;
        }
    }
    for (int i = 0; i < 1024; ++i) {
        img(i, 40) = white;
        img(i, 41) = white;
        img(i, 42) = white;
        img(i, 43) = white;
        img(i, 44) = white;
    }
    img.writeToFile("test.bmp");
    return 0;
}

