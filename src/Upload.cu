#include "Upload.hpp"

static void handleCudaError(cudaError_t error, const char* file, int line) {
	if(error != cudaSuccess) {
		char errorString[1000];
		snprintf(errorString, 1000, "FATAL ERROR (CUDA): %s in %s at line %d!\n", cudaGetErrorString(error), file, line);
		perror(errorString);
		exit(EXIT_FAILURE);
	}
}

Uploader::Uploader(std::array<Gainmap, 3> gain, std::array<Pedestalmap, 3> pedestal, std::size_t dimX, std::size_t dimY) 
  : gain(gain), pedestal(pedestal), dimX(dimX), dimY(dimY), dataBuffer(RINGBUFFER_SIZE), 
	photonBuffer(RINGBUFFER_SIZE), 
	devices(std::size_t(2 * cudaGetDeviceCount())), 
	resources(devices.size()){
	initGPUs();
	//TODO: init pedestal maps
	currentBlock.reserve(GPU_FRAMES);
}

Uploader::~Uploader() {
	freeGPUs();
}

bool Uploader::upload(std::vector<Datamap> data) {
	std::size_t i = 0;
	while(i < data.size()) {
		currentBlock.push_back(data[i]);
		if(currentBlock.size() == GPU_FRAMES) {
			//input_buffer.push(current_block);
			if(!calcFrames(currentBlock))
				return false;
			currentBlock.clear();
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
	std::vector<Gainmap> gain_unsplitted(gain.begin(), gain.end());
	std::vector<Pedestalmap> pedestal_unsplitted(pedestal.begin(), pedestal.end());
	std::vector<std::vector<Gainmap> > gain_splitted = splitMaps<Gainmap>(gain_unsplitted, devices.size());
	std::vector<std::vector<Pedestalmap> > pedestal_splitted = splitMaps<Pedestalmap>(pedestal_unsplitted, devices.size());

	if(gain_splitted.empty() || pedestal_splitted.empty()) {
			fputs("FATAL ERROR (Maps): Unexpected size!", stderr);
			exit(EXIT_FAILURE);
	}

	for(std::size_t i = 0; i < devices.size(); ++i) {
		devices[i].gain_host.push_back(gain_splitted[i][0]);
		devices[i].gain_host.push_back(gain_splitted[i][1]);
		devices[i].gain_host.push_back(gain_splitted[i][2]);

		devices[i].pedestal_host.push_back(pedestal_splitted[i][0]);
		devices[i].pedestal_host.push_back(pedestal_splitted[i][1]);
		devices[i].pedestal_host.push_back(pedestal_splitted[i][2]);

		HANDLE_CUDA_ERROR(cudaSetDevice(i / 2));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].gain, dimX * dimY * sizeof(double)));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].pedestal, dimX * dimY * sizeof(uint16_t)));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].data, dimX * dimY * sizeof(uint16_t)));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].photons, dimX * dimY * sizeof(float)));
		HANDLE_CUDA_ERROR(cudaStreamCreate(&devices[i].str));

		uploadGainmap(devices[i]);
		uploadPedestalmap(devices[i]);

		if(!resources.push(&devices[i])) {
			fputs("FATAL ERROR (RingBuffer): Unexpected size!", stderr);
			exit(EXIT_FAILURE);
		}
	}
}

void Uploader::freeGPUs() {
	for(std::size_t i = 0; i < devices.size(); ++i) {
		HANDLE_CUDA_ERROR(cudaSetDevice(devices[i].device));
		HANDLE_CUDA_ERROR(cudaFree(devices[i].gain));
		HANDLE_CUDA_ERROR(cudaFree(devices[i].pedestal));
		HANDLE_CUDA_ERROR(cudaFree(devices[i].data));
		HANDLE_CUDA_ERROR(cudaFree(devices[i].photons));
		HANDLE_CUDA_ERROR(cudaStreamDestroy(devices[i].str));
	}
}

void Uploader::uploadGainmap(struct deviceData stream) {
	HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	HANDLE_CUDA_ERROR(cudaMemcpy(stream.gain, stream.gain_host.data(), stream.gain_host.at(0).getSizeBytes() * 3, cudaMemcpyHostToDevice));
}

void Uploader::uploadPedestalmap(struct deviceData stream) {
	HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	HANDLE_CUDA_ERROR(cudaMemcpy(stream.pedestal, stream.pedestal_host.data(), stream.pedestal_host.at(0).getSizeBytes() * 3, cudaMemcpyHostToDevice));
}

void Uploader::downloadGainmap(struct deviceData stream) {
	//TODO: fix multiple gpus; completely broken
/*
	HANDLE_CUDA_ERROR(cudaSetDevice(device));
	HANDLE_CUDA_ERROR(cudaMemcpy(gain->data(), gain_device, gain->getSizeBytes(), cudaMemcpyDeviceToHost));*/
}

void Uploader::downloadPedestalmap(struct deviceData stream) {
	//TODO: fix multiple gpus; completely broken
	/*
	HANDLE_CUDA_ERROR(cudaSetDevice(device));
	HANDLE_CUDA_ERROR(cudaMemcpy(pedestal->data(), pedestal_device, pedestal->getSizeBytes(), cudaMemcpyDeviceToHost));*/
}

bool Uploader::calcFrames(std::vector<Datamap>& data) {
	//TODO: only use every second device
	std::vector<Photonmap> photonMaps;
	photonMaps.reserve(GPU_FRAMES);

	std::vector<std::vector<Datamap> > data_splitted = splitMaps<Datamap>(data, devices.size());

	for(std::size_t i = 0; i < devices.size(); ++i) {
		if(!uploadToGPU(devices[i], data[i]))
			return false;
		calculate<<<dimX, dimY, 0, devices[i].str>>>(devices[i].pedestal, devices[i].gain, devices[i].data, GPU_FRAMES, devices[i].photons);
		CHECK_CUDA_KERNEL();
		downloadFromGPU(devices[i], photonMaps);
	}

	return true;
}


bool Uploader::uploadToGPU(struct deviceData& dev, std::vector<Datamap>& data) {
	if(data.empty())
		return false;

	HANDLE_CUDA_ERROR(cudaSetDevice(dev.device));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev.data, data.data(), data.size() * sizeof(data[0]), cudaMemcpyHostToDevice, dev.str));

	return true;
}

void Uploader::downloadFromGPU(struct deviceData& dev, std::vector<Photonmap>& data) {
	std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
	//TODO: find a better way than malloc
	float* photonData = (float*)malloc(numPhotons * sizeof(float));
	if(!photonData) {
		fputs("FATAL ERROR (Memory): Allocation failed!", stderr);
		exit(EXIT_FAILURE);
	}

	HANDLE_CUDA_ERROR(cudaMemcpyAsync(photonData, dev.photons, numPhotons * sizeof(float), cudaMemcpyDeviceToHost));

	for(size_t i = 0; i < numPhotons; i += dimX * dimY) {
		data.emplace_back(dimX, dimY, &photonData[i]);
	}
}

