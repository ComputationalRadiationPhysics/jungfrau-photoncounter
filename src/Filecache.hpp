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
	//allocate space
    auto fileSize = getFileSize(path);
    auto mapSize = header ? (MAPSIZE * 2 + 32) : (MAPSIZE * 2);
    std::size_t numMaps = fileSize / static_cast<unsigned>(mapSize);

	if(fileSize + bufferPointer >= buffer.get() + sizeBytes)
	{
		std::cerr << "Error: Not enough memory allocated!\n";
		exit(EXIT_FAILURE);
	}

	//load file content
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary);
	if(!file.is_open())
	{
		std::cerr << "Error: Couldn't open file " << path << "!\n";
		exit(EXIT_FAILURE);
	}
	
    file.read(bufferPointer, fileSize);
    file.close();

    Maps<TData, TAlpaka> maps{};
    maps.numMaps = static_cast<unsigned>(numMaps); 
    maps.header = header;

	//allocate alpaka memory
    maps.data = alpaka::mem::buf::alloc<TData, typename TAlpaka::Size>(
        alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
        (numMaps * 2));

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED 
#if (SHOW_DEBUG == false)
    alpaka::mem::buf::pin(maps.data);
#endif
#endif

    alpaka::stream::StreamCpuSync streamBuf = 
        alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u);

    TData* dataBuf = reinterpret_cast<TData*>(bufferPointer);

	//copy data into alpaca memory
    alpaka::mem::view::copy(
        streamBuf,
        maps.data,
        alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                        TData,
                                        typename TAlpaka::Dim,
                                        typename TAlpaka::Size>(
            dataBuf,
            alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
            numMaps),
        numMaps);

    bufferPointer += fileSize;

    return maps;
}
