#pragma once
#include <cinttypes>
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>

namespace Bitmap {

// DWORD = uint32_t
// WORD = uint16_t
// LONG = int32_t

enum { BI_RGB, BI_RLE4, BI_RLE8, BI_BITFIELDS };

struct FileHeader {
    unsigned char bfType[2]; /* file type, must be BM */
    uint32_t bfSize;         /* size in bytes of bitmap file */
    uint16_t bfReserved1;    /* reserved, must be 0 */
    uint16_t bfReserved2;    /* reserved, must be 0 */
    uint32_t bOffBits;       /* offset in bytes from beginning of file header to
                                bitmap bits */
};

struct InfoHeader {
    uint32_t biSize;         /* number of bytes required by structure */
    int32_t biWidth;         /* width of bitmap in pixels, negative = topdown */
    int32_t biHeight;        /* height of bitmap in pixels, negative = topdown*/
    uint16_t biPlanes;       /* number of planes, must be 1 */
    uint16_t biBitCount;     /* number of bits per pixel */
    uint32_t biCompression;  /* type of compression */
    uint32_t biSizeImage;    /* size of image, 0 for BI_RGB */
    int32_t biXPelsPerMeter; /* horizontal pixel per meter */
    int32_t biYPelsPerMeter; /* vertical pixel per meter */
    uint32_t biClrUsed;      /* number of indexes used in color table */
    uint32_t biClrImportant; /* required number of color indexes */
};

struct Rgb {
    unsigned char rgbBlue;
    unsigned char rgbGreen;
    unsigned char rgbRed;
};

struct FileHeader createFileHeader();
struct InfoHeader createInfoHeader(int width, int height);

class Image {
private:
    int width;
    int height;
    FileHeader fheader;
    InfoHeader iheader;
    std::vector<Rgb> pixels;

public:
    Image(int width, int height)
        : width(width), height(height), fheader(createFileHeader()),
          iheader(createInfoHeader(width, height)), pixels(width * height)
    {
    }
    Rgb& operator()(int x, int y) { return pixels[y * width + x]; }
	void writeToFile(const std::string& path);
};

} /* namespace Bitmap */
