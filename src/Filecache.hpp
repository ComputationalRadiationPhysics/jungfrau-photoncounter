#pragma once

#include "Config.hpp"

#include <fstream>
#include <memory>


class Filecache {
private:
    std::unique_ptr<char> buffer;
    char* bufferPointer;
    const std::size_t sizeBytes;
    auto getFileSize(const std::string path) const -> off_t;

public:
    Filecache(std::size_t size);
    template <typename TData>
    auto loadMaps(const std::string& path, bool header = false) -> Maps<TData>;
};

template <typename TData>
auto Filecache::loadMaps(const std::string& path, bool header) -> Maps<TData>
{
    auto fileSize = getFileSize(path);
    auto mapSize = sizeof(TData) * MAPSIZE + (header ? FRAME_HEADER_SIZE : 0);
    auto numMaps = fileSize / static_cast<unsigned>(mapSize);

    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    file.read(bufferPointer, fileSize);
    file.close();

    Maps<TData> maps{static_cast<unsigned>(numMaps),
                     reinterpret_cast<TData*>(bufferPointer),
                     header};

    bufferPointer += fileSize;

    return maps;
}
