#pragma once

#include "RingBuffer.hpp"
#include "Pixelmap.hpp"
#include "Kernel.hpp"
#include <cmath>

#define HANDLE_CUDA_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))
#define CHECK_CUDA_KERNEL (handleCudaError(cudaGetLastError(), __FILE__, __LINE__))

const std::size_t RINGBUFFER_SIZE = 1000;
const std::size_t GPU_FRAMES = 2000000;

static void handleCudaError(cudaError_t error, const char* file, int line);

struct deviceData {
	int device;
	cudaStream_t str;
	//TODO: define types somewhere
	double* gain;
	uint16_t* pedestal;
	uint16_t* data;
	std::vector<Gainmap> gain_host;
	std::vector<Pedestalmap> pedestal_host;
	//TODO: define photon data type
	uint16_t* photons;
};

class Uploader {
public:
	Uploader(std::array<Gainmap, 3> gain, std::array<Pedestalmap, 3> pedestal, std::size_t dimX, std::size_t dimY);
	Uploader(const Uploader& other) = delete;
	Uploader& operator=(const Uploader& other) = delete;
	~Uploader();

	bool upload(std::vector<Datamap> data);
	std::vector<Photonmap> download();
protected:
private:
	RingBuffer<std::vector<Photonmap> > dataBuffer;
	//TODO: remove below (after all depencies are cleared)
	RingBuffer<deviceData*> resources;
	RingBuffer<std::vector<Photonmap> > photonBuffer;
	std::vector<Datamap> currentBlock;
	//TODO: remove below (after all depencies are cleared)
	std::array<Gainmap, 3> gain;
	//TODO: remove below (after all depencies are cleared)
	std::array<Pedestalmap, 3> pedestal;
	std::size_t dimX, dimY;
	std::vector<deviceData> devices;

	template <typename MapType>
	std::vector<std::vector<MapType> > splitMaps(std::vector<MapType>& maps, std::size_t numberOfSplits);

	void initGPUs();
	void freeGPUs();

	void uploadGainmap(struct deviceData stream);
	void uploadPedestalmap(struct deviceData stream);

	void downloadGainmap(struct deviceData stream);
	void downloadPedestalmap(struct deviceData stream);

	//OPTIONAL: add function to completely download and reassamble pedestal and Gainmaps
	//OPTIONAL: implement memory counter to prevent too much data in memory
	//OPTIONAL: implement error handling
	//OPTIONAL: make GPU_FRAMES dynamic

	bool calcFrames(std::vector<Datamap>& data);
	bool uploadToGPU(struct deviceData& dev, std::vector<Datamap>& data);
	void downloadFromGPU(struct deviceData& dev, std::vector<Photonmap>& data);
};

template<typename MapType> std::vector<std::vector<MapType> > Uploader::splitMaps(std::vector<MapType>& maps, std::size_t numberOfSplits) {
	DEBUG("splitMaps()");
	//TODO: by numberOfSplits i mean actually number of parts
	std::vector<std::vector<MapType> > ret(numberOfSplits);
	std::size_t elementsPerMap = dimX * dimY;
	std::size_t newMapSize = std::size_t(std::ceil(float(elementsPerMap) / float(numberOfSplits)));
	DEBUG("STL containers initialized!");
	//TODO: find something better than malloc!
	typename MapType::contentT* data = (typename MapType::contentT*)malloc(MapType::elementSize * newMapSize * maps.size());
	if(!data) {
		fputs("FATAL ERROR (Memory): Allocation failed!", stderr);
		exit(EXIT_FAILURE);
	}
	DEBUG("Memory for split allocated!");

	for(std::size_t i = 0; i < numberOfSplits; ++i) {
		for(std::size_t j = 0; j < maps.size(); ++j) {
			for(std::size_t k = 0; k < newMapSize; ++k) {
				data[k + i * newMapSize + j * maps.size() * elementsPerMap] = maps[j](k * newMapSize, i);
			}
			ret[i].emplace_back(newMapSize, 1, reinterpret_cast<typename MapType::contentT*>(&data[i * newMapSize + j * maps.size()]));
		}
	}
	DEBUG("splitMaps done!");
	return ret;
}
