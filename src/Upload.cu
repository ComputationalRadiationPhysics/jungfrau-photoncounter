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
	//TODO: handle pointer for multiple gpus here
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&gain_device, gain->getSizeBytes()));
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&pedestal_device, pedestal->getSizeBytes()));
	//TODO: find a better solution than this ugly hack below (using pedestal->getSizeBytes())!
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&data_device, pedestal->getSizeBytes()));
	//TODO: USE PROPER OUTPUTMAP TYPE!!!!!!!!!!
	HANDLE_CUDA_ERROR(cudaMalloc((void**)&photons_device, pedestal->getSizeBytes() * 2));
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
	//TODO: use ringbuffer of threads???
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

void Uploader::calcFrames(std::vector<Datamap>& data) {
	cudaStream_t str;
	//TODO: find something better than malloc here???
	float* photons = malloc();
	HANDLE_CUDA_ERROR(cudaCreateStream(&str));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(data_device, data->data(), data->size() * sizeof(*data[0]), cudaMemcpyHostToDevice, str));

	//TODO: use barrier or something similar here
	calculate<<<1/*blocks*/, 2/*threads*/, 0, str>>>(pedestal_device, gain_device, data_device, GPU_FRAMES, photons_device);

	//TODO: USE PROPER RETURN MAP TYPE!!!!
	HANDLE_CUDA_ERROR(cudaMemcpy(photons_device, photons, data->size() * sizeof(photons_device[0])));
	for(size_t i = 0; i < data->size(); ++i) {
		//TODO: copy data to correct maptype
	}
}

/*
void Uploader::uploadToGPU() {
	std::vector<Datamap> to_upload;
	cudaStream_t str;
	input_buffer.pop(to_upload);
	//TODO: better error handling!
	if(to_upload.empty())
		return;
	//TODO: figure out how to multithread this properly!
	HANDLE_CUDA_ERROR(cudaStreamCreate(&str));
	streams.push_back(str);
	cudaMemcpyAsync(data_device, to_upload.data(), to_upload.size() * sizeof(to_upload.data()[0]), cudaMemcpyHostToDevice, str);
}

void Uploader::downloadFromGPU() {
	//TODO: figure out how to multithread this properly!
}
*/
