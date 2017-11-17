#include "Bitmap.hpp"

#include <fstream>
#include <iostream>
#include <ostream>


namespace Bitmap {

struct FileHeader createFileHeader()
{
    struct FileHeader header;
    header.bfType[0] = 'B';
    header.bfType[1] = 'M';
    header.bfSize = 0;
    header.bfReserved1 = 0;
    header.bfReserved2 = 0;
    header.bOffBits = sizeof(struct FileHeader) + sizeof(struct InfoHeader);
    return header;
}

struct InfoHeader createInfoHeader(int width, int height)
{
    struct InfoHeader header;
    header.biSize = sizeof(header);
    header.biWidth = width;
    /* height positive for bottom-up image */
    header.biHeight = height;
    header.biPlanes = 1;
    header.biBitCount = 24;
    header.biCompression = BI_RGB;
    header.biSizeImage = 0;
    header.biXPelsPerMeter = 0;
    header.biYPelsPerMeter = 0;
    header.biClrUsed = 0;
    header.biClrImportant = 0;
    return header;
}

void Image::writeToFile(const std::string& path)
{
    std::ofstream file;
    file.open(path, std::ios::out | std::ios::binary);
    iheader.biSizeImage = width * height * sizeof(Rgb);
    fheader.bfSize =
        sizeof(FileHeader) + sizeof(InfoHeader) + width * height * sizeof(Rgb);
    file.write(reinterpret_cast<const char*>(&fheader), sizeof(fheader));
    file.write(reinterpret_cast<const char*>(&iheader), sizeof(iheader));
    file.write(reinterpret_cast<const char*>(pixels.data()),
               static_cast<signed>(pixels.size() * sizeof(Rgb)));
    file.close();
}

} /* namespace Bitmap */
