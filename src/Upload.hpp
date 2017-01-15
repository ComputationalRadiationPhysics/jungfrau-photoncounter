#pragma once

#include "RingBuffer.hpp"
#include "Pixelmap.hpp"
#include "Kernel.hpp"

#define HANDLE_CUDA_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))
#define CHECK_CUDA_KERNEL() (HANDLE_CUDA_ERROR(cudaGetLastError())

const std::size_t RINGBUFFER_SIZE = 1000;
const std::size_t GPU_FRAMES = 2000000;

static void handleCudaError(cudaError_t error, const char* file, int line);

struct deviceData {
	int device;
	cudaStream_t str;
	double* gain;
	uint16_t* pedestal;
	uint16_t* data;
	float* photons;
};

class Uploader {
public:
	Uploader(Gainmap gain, Pedestalmap pedestal, std::size_t dimX, std::size_t dimY);
	Uploader(const Uploader& other) = delete;
	Uploader& operator=(const Uploader& other) = delete;
	~Uploader();

	bool upload(std::vector<Datamap> data);
	std::vector<Datamap> download();
protected:
private:
	RingBuffer<std::vector<Photonmap> > dataBuffer;
	RingBuffer<deviceData&> resources;
	RingBuffer<std::vector<Photonmap> > photonBuffer;
	std::vector<Datamap> currentBlock;
	Gainmap gain;
	Pedestalmap pedestal;
	std::size_t dimX, dimY;
	std::vector<deviceData> devices;

	void initGPUs();
	void freeGPUs();

	void uploadGainmap(struct device stream);
	void uploadPedestalmap(struct device stream);

	void downloadGainmap(struct device stream);
	void downloadPedestalmap(struct device stream);

	//OPTIONAL: add function to completely download and reassamble pedestal and Gainmaps
	//OPTIONAL: implement memory counter to prevent too much data in memory
	//OPTIONAL: implement error handling
	//OPTIONAL: make GPU_FRAMES dynamic

	bool Uploader::calcFrames(std::vector<Datamap>& data);
	bool uploadToGPU(std::vector<Datamap>& data, struct deviceData& dev);
	std::vector<Photonmap>& downloadFromGPU(struct deviceData& dev);
}
