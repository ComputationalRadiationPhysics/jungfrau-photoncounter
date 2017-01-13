#include "upload.hpp"

static void handleCudaError(cudaError_t error, const char* file, int line) {
	if(err != cudaSuccess) {
		char errorString[1000];
		snprintf(errorString, 1000, "%s in %s at line %d!\n", cudaGetErrorString(error), file, line);
		perror(errorString);
		exit(EXIT_FAILURE);
	}
}

Uploader::Uploader(Gainmap& gain, Pedestalmap& pedastel) : gain(gain), pedestal(pedestal), input_buffer(RINGBUFFER_SIZE), output_buffer(RINGBUFFER_SIZE), mem_usage(0){
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&gain_device, gain->getSizeBytes()));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pedestal_device, pedestal->getSizeBytes()));
	//TODO: find a better solution than this ugly hack below (using pedestal->getSizeBytes())!
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&data_device, pedestal->getSizeBytes()));
	uploadGainmap(*gain);
	uploadPedestalmap(*pedestal);
	current_block.reserve(GPU_FRAMES);
}

Uploader::~Uploader() {
	HANDLE_CUDA_ERROR(cudaFree(gain_device));
	HANDLE_CUDA_ERROR(cudaFree(pedestal_device));
}

void Uploader::upload(std::vector<Datamap> data) {
	size_t i = 0;
	while(i < data.size()) {
		current_block.push_back(data[i]);
		if(current_block.size() == GPU_FRAMES) {
			input_buffer.push(current_block);
			current_block.clear();
		}
	}
}

std::vector<Datamap> Uploader::download() {
	std::vector<Datamap> ret;
	output_buffer.pop(ret);
	return ret;
}

void Uploader::uploadGainmap() {
	HANDLE_CUDA_ERROR(cudaMemcpy(gain_device, gain->data(), gain->getSizeBytes(), cudaMemcpyHostToDevice));
}

void Uploader::uploadPedestalmap() {
	HANDLE_CUDA_ERROR(cudaMemcpy(pedestal_device, pedestal->data(), pedestal->getSizeBytes(), cudaMemcpyHostToDevice));
}

void Uploader::downloadGainmap() {
	HANDLE_CUDA_ERROR(cudaMemcpy(gain->data(), gain_device, gain->getSizeBytes(), cudaMemcpyDeviceToHost));
}

void Uploader::downloadPedestalmap() {
	HANDLE_CUDA_ERROR(cudaMemcpy(pedestal->data(), pedestal_device, pedestal->getSizeBytes(), cudaMemcpyDeviceToHost));
}

void Uploader::uploadToGPU() {
	std::vector<Datamap> to_upload;
	input_buffer.pop(to_upload);
	//TODO: better error handling!
	if(to_upload.empty())
		return;
	//TODO: figure out how to multithread this properly!

}

void Uploader::downloadFromGPU() {
	//TODO: figure out how to multithread this properly!
}
