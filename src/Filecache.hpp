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
    template <typename TData, typename TAlpaka>
    auto loadMaps(const std::string& path, bool header = false)
        -> Maps<TData, TAlpaka>;
};

template <typename TData, typename TAlpaka>
auto Filecache::loadMaps(const std::string& path, bool header)
    -> Maps<TData, TAlpaka>
{
    auto fileSize = getFileSize(path);
    auto mapSize = sizeof(TData) * MAPSIZE + (header ? FRAME_HEADER_SIZE : 0);
    auto numMaps = fileSize / static_cast<unsigned>(mapSize);

    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
    file.read(bufferPointer, fileSize);
    file.close();

    Maps<TData, TAlpaka> maps{};
    maps.numMaps = static_cast<unsigned>(numMaps); 
    maps.header = header;

    maps.data = alpaka::mem::buf::alloc<TData, typename TAlpaka::Size>(
        alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
        numMaps * (MAPSIZE + (header ? FRAMEOFFSET : 0)));
#if (SHOW_DEBUG == false)
    alpaka::mem::buf::pin(maps.data);
#endif

    alpaka::stream::StreamCpuSync streamBuf = 
        alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u);

    TData* dataBuf = reinterpret_cast<TData*>(bufferPointer);

    alpaka::mem::view::copy(
        streamBuf,
        maps.data,
        alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                        TData,
                                        typename TAlpaka::Dim,
                                        typename TAlpaka::Size>(
            dataBuf,
            alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
            (numMaps * (MAPSIZE + (header ? FRAMEOFFSET : 0)))),
        numMaps * (MAPSIZE + (header ? FRAMEOFFSET : 0)));

    bufferPointer += fileSize;

    return maps;
}
