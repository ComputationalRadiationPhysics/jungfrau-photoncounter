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
	//devices(std::size_t(2 * cudaGetDeviceCount())), 
	resources(100){
	DEBUG("Entering Uploader cunstructor!");
	int num = 0;
	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&num));
	DEBUG("# of CUDA devices: " << num);
	//TODO: fix multi GPU
	DEBUG("FIXME: forcing to use only one GPU!");
	num = 1;

	devices.resize(2 * num);
	DEBUG("Initializing GPUs!");
	initGPUs();
	//TODO: init pedestal maps
	DEBUG("Reserving memory for currentBlock.");
	currentBlock.reserve(GPU_FRAMES);
	DEBUG("End of constructor!");
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
	//photonBuffer.pop(ret);
	return ret;
}

void Uploader::initGPUs() {
	DEBUG("initGPU()");
	std::vector<Gainmap> gain_unsplitted(gain.begin(), gain.end());
	std::vector<Pedestalmap> pedestal_unsplitted(pedestal.begin(), pedestal.end());
	DEBUG("Maps created");
	std::vector<std::vector<Gainmap> > gain_splitted = splitMaps<Gainmap>(gain_unsplitted, devices.size());
	std::vector<std::vector<Pedestalmap> > pedestal_splitted = splitMaps<Pedestalmap>(pedestal_unsplitted, devices.size());
	DEBUG("Maps splitted");

	if(gain_splitted.empty() || pedestal_splitted.empty()) {
			fputs("FATAL ERROR (Maps): Unexpected size!", stderr);
			exit(EXIT_FAILURE);
	}

	DEBUG("# of maps: " << gain_splitted.size() << " : " << pedestal_splitted.size());

	for(std::size_t i = 0; i < 1/*devices.size()*/; ++i) {
		DEBUG("Uploading Pedestalmaps for device " << i / 2 << "with i=" << i);
		devices[i].gain_host.push_back(gain_splitted[i][0]);
		devices[i].gain_host.push_back(gain_splitted[i][1]);
		devices[i].gain_host.push_back(gain_splitted[i][2]);

		DEBUG("Uploading Gainmaps for device " << i / 2 << "with i=" << i);
		devices[i].pedestal_host.push_back(pedestal_splitted[i][0]);
		devices[i].pedestal_host.push_back(pedestal_splitted[i][1]);
		devices[i].pedestal_host.push_back(pedestal_splitted[i][2]);

		DEBUG("Setting device " << i / 2);
		HANDLE_CUDA_ERROR(cudaSetDevice(i / 2));
		DEBUG("Allocating GPU memory on device for #" << i);
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].gain, dimX * dimY * sizeof(double)));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].pedestal, dimX * dimY * sizeof(uint16_t)));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].data, dimX * dimY * sizeof(uint16_t)));
		HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].photons, dimX * dimY * sizeof(uint16_t)));
		HANDLE_CUDA_ERROR(cudaStreamCreate(&devices[i].str));

		DEBUG("Uploading Gainmaps for #" << i);
		uploadGainmap(devices[i]);
		DEBUG("Uploading Pedestalmaps for #" << i);
		uploadPedestalmap(devices[i]);

		if(!resources.push(&devices[i])) {
			fputs("FATAL ERROR (RingBuffer): Unexpected size!", stderr);
			exit(EXIT_FAILURE);
		}
	}
	DEBUG("initGPU done!");
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
	DEBUG("Gainmap upload ...");
	HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	HANDLE_CUDA_ERROR(cudaMemcpy(stream.gain, stream.gain_host.data(), stream.gain_host.at(0).getSizeBytes() * 3, cudaMemcpyHostToDevice));
	DEBUG("Done!");
}

void Uploader::uploadPedestalmap(struct deviceData stream) {
	DEBUG("Pedestalmap upload ...");
	HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	HANDLE_CUDA_ERROR(cudaMemcpy(stream.pedestal, stream.pedestal_host.data(), stream.pedestal_host.at(0).getSizeBytes() * 3, cudaMemcpyHostToDevice));
	DEBUG("Done!");
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
		if(!uploadToGPU(devices[i], data_splitted[i]))
			return false;
		calculate<<<devices.size() / 128, 128, (3 * sizeof(uint16_t) + 3 * sizeof(double)) * 128, devices[i].str>>>(uint16_t(dimX * dimY / devices.size()), devices[i].pedestal, devices[i].gain, devices[i].data, uint16_t(GPU_FRAMES), devices[i].photons);
		CHECK_CUDA_KERNEL;
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
	uint16_t* photonData = (uint16_t*)malloc(numPhotons * sizeof(uint16_t));
	if(!photonData) {
		fputs("FATAL ERROR (Memory): Allocation failed!", stderr);
		exit(EXIT_FAILURE);
	}

	HANDLE_CUDA_ERROR(cudaMemcpyAsync(photonData, dev.photons, numPhotons * sizeof(uint16_t), cudaMemcpyDeviceToHost));

	for(size_t i = 0; i < numPhotons; i += dimX * dimY) {
		data.emplace_back(dimX, dimY, &photonData[i]);
	}
}

