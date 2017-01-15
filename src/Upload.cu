#include "upload.hpp"

static void handleCudaError(cudaError_t error, const char* file, int line) {
	if(err != cudaSuccess) {
		char errorString[1000];
		snprintf(errorString, 1000, "FATAL ERROR (CUDA): %s in %s at line %d!\n", cudaGetErrorString(error), file, line);
		perror(errorString);
		exit(EXIT_FAILURE);
	}
}

Uploader::Uploader(Gainmap gain, Pedestalmap pedestal, std::size_t dimX, std::size_t dimY) : gain(gain), pedestal(pedestal), dimX(dimX), dimY(dimY), dataBuffer(RINGBUFFER_SIZE), photonBuffer(RINGBUFFER_SIZE), devices(2 * cudaGetDeviceCount()), resources(devices.size()){
	initGPUs(devices.size());
	//TODO: init pedestal maps
	current_block.reserve(GPU_FRAMES);
}

Uploader::~Uploader() {
	freeGPUs();
}

bool Uploader::upload(std::vector<Datamap> data) {
	std::size_t i = 0;
	while(i < data.size()) {
		current_block.push_back(data[i]);
		if(current_block.size() == GPU_FRAMES) {
			//input_buffer.push(current_block);
			if(!calcFrames(current_block))
				return false;
			current_block.clear();
		}
	}
	return true;
}

std::vector<Photonmap> Uploader::download() {
	std::vector<Photonmap> ret;
	photonBuffer.pop(ret);
	return ret;
}

void Uploader::initGPUs() {
	for(std::size_t i = 0; i < devices.size(); ++i) {
		HANDLE_CUDA_ERROR(cudaSetDevice(i / 2));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].gain, dimX * dimY * sizeof(double)));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].pedestal, dimX * dimY * sizeof(uint16_t)));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].data, dimX * dimY * sizeof(uint16_t)));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].photons, dimX * dimY * sizeof(float)));
		HANDLE_CUDA_ERROR(cudaCreateStream(&devices[i].str));

		uploadGainmap(i);
		uploadPedestalmap(i);

		if(!resources.push(devices[i])) {
			fputs("FATAL ERROR (RingBuffer): Unexpected size!", stderr);
			exit(EXIT_FAILURE);
		}
	}
}

void Uploader::freeGPUs() {
	for(std::size_t i = 0; i < devices.size(); ++i) {
		HANDLE_CUDA_ERROR(cudaSetDevice(i));
		HANDLE_CUDA_ERROR(cudaFree(device[i].gain));
		HANDLE_CUDA_ERROR(cudaFree(device[i].pedestal));
		HANDLE_CUDA_ERROR(cudaFree(device[i].data));
		HANDLE_CUDA_ERROR(cudaFree(device[i].photons));
		HANDLE_CUDA_ERROR(cudaStreamDestroy(devices[i].in));
		HANDLE_CUDA_ERROR(cudaStreamDestroy(devices[i].out));
	}
}

void Uploader::uploadGainmap(struct device stream) {
	//TODO: fix multiple gpus
	HANDLE_CUDA_ERROR(cudaSetDevice(device));
	HANDLE_CUDA_ERROR(cudaMemcpy(gain_device, gain->data(), gain->getSizeBytes(), cudaMemcpyHostToDevice));
}

void Uploader::uploadPedestalmap(struct device stream) {
	//TODO: fix multiple gpus
	HANDLE_CUDA_ERROR(cudaSetDevice(device));
	HANDLE_CUDA_ERROR(cudaMemcpy(pedestal_device, pedestal->data(), pedestal->getSizeBytes(), cudaMemcpyHostToDevice));
}

void Uploader::downloadGainmap(struct device stream) {
	//TODO: fix multiple gpus
	HANDLE_CUDA_ERROR(cudaSetDevice(device));
	HANDLE_CUDA_ERROR(cudaMemcpy(gain->data(), gain_device, gain->getSizeBytes(), cudaMemcpyDeviceToHost));
}

void Uploader::downloadPedestalmap(struct device stream) {
	//TODO: fix multiple gpus
	HANDLE_CUDA_ERROR(cudaSetDevice(device));
	HANDLE_CUDA_ERROR(cudaMemcpy(pedestal->data(), pedestal_device, pedestal->getSizeBytes(), cudaMemcpyDeviceToHost));
}

bool Uploader::calcFrames(std::vector<Datamap>& data) {
	//TODO: fix multiple gpus
	struct deviceData *dev;
	std::vector<Photonmap> photonMaps;
	photonMaps.reserve(GPU_FRAMES);

	if(resources.pop(dev))
		return false;

	if(!uploadToGPU(*dev, *data))
		return false;
	//TODO: use barrier or something similar here
	CHECK_CUDA_KERNEL(calculate<<<dimX, dimY, 0, dev.str>>>(dev.pedestal, dev.gain, dev.data, GPU_FRAMES, dev.photon));
	photonMaps = downlodFromGPU(*dev);

	if(resources.push(dev)) {
		fputs("FATAL ERROR (RingBuffer): Unexpected size!", stderr);
		exit(EXIT_FAILURE);
	}

	return true;
}


bool Uploader::uploadToGPU(struct deviceData& dev, std::vector<Datamap>& data) {
	if(data.empty())
		return false;

	HANDLE_CUDA_ERROR(cudaSetDevice(dev.device));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev.data, data->data(), data->size() * sizeof(*data[0]), cudaMemcpyHostToDevice, dev.str));

	return true;
}

std::vector<Photonmap> Uploader::downloadFromGPU(struct deviceData& dev) {
	std::vector<Photonmap> data;
	std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
	//TODO: find a better way than malloc
	float* photonData = malloc(numPhotons * sizeof(float));
	if(!photonData) {
		fputs("FATAL ERROR (Memory): Allocation failed!", stderr);
		exit(EXIT_FAILURE);
	}

	HANDLE_CUDA_ERROR(cudaMemcpyAsync(photonData, dev.photons, numPhotons * sizeof(float)));

	for(size_t i = 0; i < numPhotons; i += dimX * dimY) {
		data.emplace_back(dimX, dimY, &photonData[i]);
	}

	return data;
}

