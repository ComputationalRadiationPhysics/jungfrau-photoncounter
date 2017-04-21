#pragma once
#include "Pixelmap.hpp"
#include <fstream>
#include <memory>

class Filecache {
private:
    std::unique_ptr<char, decltype(cudaFreeHost)*> buffer;
    char* bufferPointer;
    const std::size_t sizeBytes;
    off_t getFileSize(const std::string path) const;
	bool header;

public:
    Filecache(std::size_t sizeBytes, bool header = false);
    template <typename Maptype> Maptype loadMaps(const std::string& path);
};

template <typename Maptype> Maptype Filecache::loadMaps(const std::string& path)
{
    auto fileSize = getFileSize(path);
    auto mapSize = Maptype::elementSize * DIMX * DIMY + (header ? FRAME_HEADER_SIZE : 0);
    auto numMaps = fileSize / mapSize;

    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    file.read(bufferPointer, fileSize);
    file.close();

        Maptype maps(numMaps, reinterpret_cast<typename Maptype::contentT*>(bufferPointer)));

        return maps;
}
