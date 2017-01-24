#pragma once
#include <cinttypes>

namespace Bitmap {

// DWORD = uint32_t
// WORD = uint16_t
// LONG = int32_t

enum { BI_RGB, BI_RLE4, BI_RLE8, BI_BITFIELDS };

struct FileHeader {
    uint16_t bfType;      /* file type, must be BM */
    uint32_t bfSize;      /* size in bytes of bitmap file */
    uint16_t bfReserved1; /* reserved, must be 0 */
    uint16_t bfReserved2; /* reserved, must be 0 */
    uint32_t bOffBits;    /* offset in bytes from beginning of file header to
                             bitmap bits */
};

struct InfoHeader {
    uint32_t biSize;         /* number of bytes required by structure */
    int32_t biWidth;         /* width of bitmap in pixels, negative = topdown */
    int32_t biHeight;        /* height of bitmap in pixels negative = topdown*/
    uint16_t biPlanes;       /* number of planes, must be 1 */
    uint16_t biBitCount;     /* number of bits per pixel */
    uint32_t biCompression;  /* type of compression */
    uint32_t biSizeImage;    /* size of image, 0 for BI_RGB */
    int32_t biXPelsPerMeter; /* horizontal pixel per meter */
    int32_t biYPelsPerMeter; /* vertical pixel per meter */
    uint32_t biClrUsed;      /* number of indexes used in color table */
    uint32_t biClrImportant; /* required number of color indexes */
};

struct RgbQuad {
    unsigned char rgbBlue;
    unsigned char rgbGreen;
    unsigned char rgbRed;
    unsigned char rgbReserved; /* must be 0 */
};

template<typename BitCountType>
class Image {
private:
	Fileheader fheader;
	InfoHeader iheader;
	std::vector<RgbQuad> colors;
	std::vector<BitCountType> pixels;
public:
	Image(int width, int height) {
	};
};

} /* namespace Bitmap */
