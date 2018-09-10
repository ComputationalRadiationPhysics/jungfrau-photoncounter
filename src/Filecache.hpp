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
  template <typename TData, typename TAlpaka, typename TDim, typename TSize>
    auto loadMaps(const std::string& path, bool header = false)
    -> FramePakage<TData, TAlpaka, TDim, TSize>;
};

template <typename TData, typename TAlpaka, typename TDim, typename TSize>
auto Filecache::loadMaps(const std::string& path, bool header)
    -> FramePakage<TData, TAlpaka, TDim, TSize>
{
	//allocate space
    auto fileSize = getFileSize(path);
    auto mapSize = header ? (MAPSIZE * 2 + 32) : (MAPSIZE * 2);
    std::size_t numFrames = fileSize / static_cast<unsigned>(mapSize);

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

    FramePakage<TData, TAlpaka, TDim, TSize> maps{};
    maps.numFrames = static_cast<unsigned>(numFrames); 

	//allocate alpaka memory
    maps.data = alpaka::mem::buf::alloc<TData, TSize>(
        alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
        (numFrames * 2));

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED 
#if (SHOW_DEBUG == false)
    alpaka::mem::buf::pin(maps.data);
#endif
#endif

    alpaka::queue::QueueCpuSync streamBuf = 
        alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u);

    TData* dataBuf = reinterpret_cast<TData*>(bufferPointer);

	//copy data into alpaca memory
    alpaka::mem::view::copy(
        streamBuf,
        maps.data,
        alpaka::mem::view::ViewPlainPtr<typename TAlpaka::DevHost,
                                        TData,
                                        TDim,
                                        TSize>(
            dataBuf,
            alpaka::pltf::getDevByIdx<typename TAlpaka::PltfHost>(0u),
            numFrames),
        numFrames);

    bufferPointer += fileSize;

    return maps;
}
