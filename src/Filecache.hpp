#pragma once
#include "Pixelmap.hpp"
#include <fstream>
#include <memory>

class Filecache {
private:
    std::unique_ptr<char[]> buffer;
    char* bufferPointer;
    const std::size_t sizeBytes;
    off_t getFileSize(const std::string path) const;

public:
    Filecache(std::size_t sizeBytes);
    template <typename Maptype>
    std::vector<Maptype> loadMaps(const std::string& path,
                                  const std::size_t dimX,
                                  const std::size_t dimY);
};

template <typename Maptype>
std::vector<Maptype> Filecache::loadMaps(const std::string& path,
                                         const std::size_t dimX,
                                         const std::size_t dimY)
{
    auto fileSize = getFileSize(path);
    auto mapSize = Maptype::elementSize * dimX * dimY;
    auto numMaps = fileSize / mapSize;
    std::vector<Maptype> maps;
    maps.reserve(numMaps);
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    file.read(bufferPointer, fileSize);
    file.close();
    for (std::size_t i = 0; i < numMaps; ++i) {
        maps.emplace_back(dimX, dimY,
                          reinterpret_cast<typename Maptype::contentT*>(bufferPointer));
        bufferPointer += mapSize;
    }
    return maps;
}
