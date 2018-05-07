#pragma once
#include <fstream>
#include <iostream>
#include <ostream>
#include <vector>

namespace Bitmap {

enum { BI_RGB, BI_RLE4, BI_RLE8, BI_BITFIELDS };

#pragma pack(push, 1)
struct FileHeader {
    unsigned char bfType[2];   /* file type, must be BM */
    std::uint32_t bfSize;      /* size in bytes of bitmap file */
    std::uint16_t bfReserved1; /* reserved, must be 0 */
    std::uint16_t bfReserved2; /* reserved, must be 0 */
    std::uint32_t bOffBits; /* offset in bytes from beginning of file header to
                          bitmap bits */
};

struct InfoHeader {
    std::uint32_t biSize;   /* number of bytes required by structure */
    std::int32_t biWidth;   /* width of bitmap in pixels, negative = topdown */
    std::int32_t biHeight;  /* height of bitmap in pixels, negative = topdown*/
    std::uint16_t biPlanes; /* number of planes, must be 1 */
    std::uint16_t biBitCount;     /* number of bits per pixel */
    std::uint32_t biCompression;  /* type of compression */
    std::uint32_t biSizeImage;    /* size of image, 0 for BI_RGB */
    std::int32_t biXPelsPerMeter; /* horizontal pixel per meter */
    std::int32_t biYPelsPerMeter; /* vertical pixel per meter */
    std::uint32_t biClrUsed;      /* number of indexes used in color table */
    std::uint32_t biClrImportant; /* required number of color indexes */
};

struct Rgb {
    unsigned char rgbBlue;
    unsigned char rgbGreen;
    unsigned char rgbRed;
};
#pragma pack(pop)

struct FileHeader createFileHeader();
struct InfoHeader createInfoHeader(int width, int height);

class Image {
private:
    std::size_t width;
    std::size_t height;
    FileHeader fheader;
    InfoHeader iheader;
    std::vector<Rgb> pixels;

public:
    Image(std::size_t widthIn, std::size_t heightIn)
        : width(widthIn),
          height(heightIn),
          fheader(createFileHeader()),
          iheader(createInfoHeader(widthIn, heightIn)),
          pixels(widthIn * heightIn)
    {
    }
    Rgb& operator()(std::size_t x, std::size_t y)
    {
        return pixels[y * width + x];
    }
    void writeToFile(const std::string& path);
};

} /* namespace Bitmap */
