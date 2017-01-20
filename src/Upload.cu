#include "Upload.hpp"

std::size_t Uploader::nextFree = 0;
std::vector<deviceData> devices;

static void handleCudaError(cudaError_t error, const char* file, int line)
{
    if (error != cudaSuccess) {
        char errorString[1000];
        snprintf(errorString, 1000,
                 "FATAL ERROR (CUDA, %d): %s in %s at line %d!\n", error,
                 cudaGetErrorString(error), file, line);
        fputs(errorString, stderr);
        exit(EXIT_FAILURE);
    }
}

Uploader::Uploader(std::array<Gainmap, 3> gain,
                   std::array<Pedestalmap, 3> pedestal, std::size_t dimX,
                   std::size_t dimY, std::size_t numberOfDevices)
    : gain(gain), pedestal(pedestal), dimX(dimX), dimY(dimY),
      resources(/*2 * numberOfDevices*/1)
 {
	 DEBUG("Entering uploader constructor!");
	 // TODO: FIX MULTI GPU
	 DEBUG("FIXME: Forcing to use only one GPU (with one stream)!");
	 //TODO: use size function of ringbuffer (implement maybe)
	 devices.resize(2 * numberOfDevices);

	 DEBUG("Initializing GPUs!");
	 initGPUs();
	 // TODO: init pedestal maps
	 DEBUG("Reserving memory for currentBlock.");
	 currentBlock.reserve(GPU_FRAMES);
	 DEBUG("End of constructor!");
 }

 Uploader::~Uploader() { freeGPUs(); }

 bool Uploader::upload(std::vector<Datamap> data)
 {
	 std::size_t i = 0;
	 while (i < data.size()) {
		 currentBlock.push_back(data[i]);
		 if (currentBlock.size() == GPU_FRAMES) {
			 // input_buffer.push(current_block);
			 if (!calcFrames(currentBlock))
				 return false;
			 currentBlock.clear();
		 }
	 }
	 return true;
 }

 std::vector<Photonmap> Uploader::download()
 {
	 std::vector<Photonmap> ret;
	 int current = nextFree;
	 if(devices[nextFree].state != READY)
		 return ret;
	 ++nextFree;

	 /*	//TODO: use local photonmap 
	 for(size_t i = 0; i < GPU_FRAMES * dimX * dimY; ++i){
			 devices[current].photon_host.emplace_back(dimX, dimY, devices[current].photon);
			 }*/

	 ret = Uploader::devices[current].photon_host;
	 Uploader::devices[current].photon_host.clear();
	 if(!resources.push(&devices[current])) {
		 fputs("FATAL ERROR (RingBuffer): Unexpected size!", stderr);
		 exit(EXIT_FAILURE);
	 }
	 return ret;
 }

 void Uploader::callback(void* data) {
	 //TODO: does this work like that
	 Uploader::devices[*((int*)data)].state = READY;
 }

 void Uploader::initGPUs()
 {
	 DEBUG("initGPU()");

	 for (std::size_t i = 0; i < devices.size(); ++i) {
		 DEBUG("Uploading Pedestalmaps for device " << i / 2 << " with i=" << i);
		 devices[i].gain_host = &gain;

		 DEBUG("Uploading Gainmaps for device " << i / 2 << " with i=" << i);
		 devices[i].pedestal_host = &pedestal;

		 devices[i].state = FREE;
		 //TODO: is this really needed? if yes, throw out device member
		 devices[i].id = i;

		 DEBUG("Setting device " << i / 2);
		 HANDLE_CUDA_ERROR(cudaSetDevice(i / 2));

		 // TODO: fix dimensions!!!
		 DEBUG("Allocating GPU memory on device for #" << i);
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].gain, dimX * dimY * sizeof(double) * 3));
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].pedestal, dimX * dimY * sizeof(uint16_t) * 3));
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].data, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].photons, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));

		 DEBUG("Creating GPU stream #" << i);
		 HANDLE_CUDA_ERROR(cudaStreamCreate(&devices[i].str));

		 synchronize();

		 DEBUG("Uploading Gainmaps for #" << i);
		 uploadGainmap(devices[i]);
		 DEBUG("Uploading Pedestalmaps for #" << i);
		 uploadPedestalmap(devices[i]);

		 if (!resources.push(&devices[i])) {
			 fputs("FATAL ERROR (RingBuffer): Unexpected size!", stderr);
			 exit(EXIT_FAILURE);
		 }
	 }
	 DEBUG("initGPU done!");
 }

 void Uploader::freeGPUs()
 {
	 synchronize();
	 for (std::size_t i = 0; i < devices.size(); ++i) {
		 HANDLE_CUDA_ERROR(cudaSetDevice(devices[i].device));
		 HANDLE_CUDA_ERROR(cudaFree(devices[i].gain));
		 HANDLE_CUDA_ERROR(cudaFree(devices[i].pedestal));
		 HANDLE_CUDA_ERROR(cudaFree(devices[i].data));
		 HANDLE_CUDA_ERROR(cudaFree(devices[i].photons));
		 HANDLE_CUDA_ERROR(cudaStreamDestroy(devices[i].str));
	 }
 }

 void Uploader::synchronize()
 {
	 for (struct deviceData dev : devices)
		 HANDLE_CUDA_ERROR(cudaStreamSynchronize(dev.str));
 }

 void Uploader::uploadGainmap(struct deviceData stream)
 {
	 DEBUG("Gainmap upload ...");
	 HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	 DEBUG("cudaMemcpy(" << stream.gain << ", " << stream.gain_host->at(0).data() << ", " << stream.gain_host->at(0).getSizeBytes() * 3 << ", cudaMemcpyHostToDevice);");
	 HANDLE_CUDA_ERROR(cudaMemcpy(stream.gain, stream.gain_host->at(0).data(), stream.gain_host->at(0).getSizeBytes() * 3, cudaMemcpyHostToDevice));
	 DEBUG("Done!");
 }

 void Uploader::uploadPedestalmap(struct deviceData stream)
 {
	 DEBUG("Pedestalmap upload ...");
	 HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	 HANDLE_CUDA_ERROR(cudaMemcpy(stream.pedestal, stream.pedestal_host->at(0).data(), stream.pedestal_host->at(0).getSizeBytes() * 3, cudaMemcpyHostToDevice));
	 DEBUG("Done!");
 }

 void Uploader::downloadGainmap(struct deviceData stream)
 {
	 DEBUG("Gainmap upload ...");
	 HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	 DEBUG("cudaMemcpy(" << stream.gain_host->at(0).data() << ", " << stream.gain << ", " << stream.gain_host->at(0).getSizeBytes() * 3 << ", cudaMemcpyHostToDevice);");
	 HANDLE_CUDA_ERROR(cudaMemcpy(stream.gain_host->at(0).data(), stream.gain, stream.gain_host->at(0).getSizeBytes() * 3, cudaMemcpyDeviceToHost));
	 DEBUG("Done!");
 }

 void Uploader::downloadPedestalmap(struct deviceData stream)
 {
	 DEBUG("Pedestalmap doanload ...");
	 HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	 HANDLE_CUDA_ERROR(cudaMemcpy(stream.pedestal_host->at(0).data(), stream.pedestal, stream.pedestal_host->at(0).getSizeBytes() * 3, cudaMemcpyDeviceToHost));
	 DEBUG("Done!");
 }

 bool Uploader::calcFrames(std::vector<Datamap>& data)
 {
	 DEBUG("calcFrames");
	 std::vector<Photonmap> photonMaps;
	 photonMaps.reserve(GPU_FRAMES);

	 if(data.empty())
		 return false;

	 struct deviceData dev;
	 if(!resources.pop(dev))
		 return false;

	 uploadToGPU(dev, data);

	 //TODO: FIX KERNEL CALL
	 calculate<<</*DEVI.size() * NODES_PER_GPU / 128*/ 1, 128, 3 * (sizeof(uint16_t) + sizeof(double)) * 128, dev.str>>>(uint16_t(dimX * dimY / devices.size()), devices[i].pedestal, devices[i].gain, devices[i].data, uint16_t(GPU_FRAMES), devices[i].photons);
	 CHECK_CUDA_KERNEL;
	 downloadFromGPU(dev);

	 DEBUG("Creating callback ...");
	 HANDLE_CUDA_ERROR(cudaStreamAddCallback(dev.str, Uploader::callback, &dev.id, 0));

	 DEBUG("End calcFrames");
	 return true;
 }

void Uploader::uploadToGPU(struct deviceData& dev, std::vector<Datamap>& data)
{
    HANDLE_CUDA_ERROR(cudaSetDevice(dev.device));
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev.data, data.data(), data.size() * sizeof(data[0]), cudaMemcpyHostToDevice, dev.str));
}

void Uploader::downloadFromGPU(struct deviceData& dev)
{
    DEBUG("Entering downloadFromGPU (str=" << dev.str << ")");
    std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
    DEBUG("numPhotons = " << numPhotons);
    // TODO: find a better way than malloc
    uint16_t* photonData = (uint16_t*)malloc(numPhotons * sizeof(uint16_t));
    if (!photonData) {
        fputs("FATAL ERROR (Memory): Allocation failed!", stderr);
        exit(EXIT_FAILURE);
    }
    DEBUG(numPhotons * sizeof(uint16_t) << " Bytes allocated");
    DEBUG("cudaMemcpyAsync(" << photonData << ", " << dev.photons << ", " << numPhotons * sizeof(uint16_t) << ", cudaMemcpyDeviceToHost, " << dev.str << ");");

    HANDLE_CUDA_ERROR(cudaSetDevice(dev.device));
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(photonData, dev.photons, numPhotons * sizeof(uint16_t), cudaMemcpyDeviceToHost, dev.str));

    DEBUG("data downloaded");

    for (size_t i = 0; i < numPhotons; i += dimX * dimY) {
        dev.photon_host.emplace_back(dimX, dimY, &photonData[i]);
    }
    DEBUG("Data written. downloadFromGPU done!");
}

