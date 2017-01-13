#pragma once

#include "RingBuffer.hpp"
#include "Pixelmap.hpp"
#include "Kernel.hpp"

#define HANDLE_CUDA_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))

const size_t RINGBUFFER_SIZE = 1000;
const size_t GPU_FRAMES = 2000000;

static void handleCudaError(cudaError_t error, const char* file, int line);

class Uploader {
public:
	Uploader(Gainmap& gain, Pedestalmap& pedastel);
	Uploader(const Uploader& other) = delete;
	Uploader& operator=(const Uploader& other) = delete;
	~Uploader();

	void upload(std::vector<Datamap> data);
	std::vector<Datamap> download();
protected:
private:
	RingBuffer<std::vector<Datamap> > output_buffer;
	//TODO: use correct map here!!
	RingBuffer<std::vector<Datamap> > input_buffer;
	std::vector<Datamap> current_block;
	Gainmap* gain;
	Pedestalmap* pedestal;

	std::vector<cudaStream_t> streams;

	//device vars
	double* gain_device;
	uint16_t* pedestal_device;
	uint16_t* data_device;
	float* photons_device;

	void uploadGainmap();
	void uploadPedestalmap();

	void downloadGainmap();
	void downloadPedestalmap();

	void uploadToGPU();
	void downloadFromGPU();
}
