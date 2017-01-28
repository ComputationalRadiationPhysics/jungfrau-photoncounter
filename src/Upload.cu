#include "Upload.hpp"

std::size_t Uploader::nextFree = 0;
std::size_t Uploader::dimX = 0;
std::size_t Uploader::dimY = 0;
std::vector<deviceData> Uploader::devices;

bool isMapEmpty(Datamap map, std::size_t dimX, std::size_t dimY) {
	for(std::size_t y = 0; y < dimY; ++y) {
		for(std::size_t x = 0; x < dimX; ++x) {
			if(map(x, y) != 0)
				return false;
		}
	}
	return true;
}

void handleCudaError(cudaError_t error, const char* file, int line)
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
    : gain(gain), pedestal(pedestal), resources(STREAMS_PER_GPU * numberOfDevices)
 {
	 Uploader::dimX = dimX;
	 Uploader::dimY = dimY;
	 DEBUG("Entering uploader constructor!");
	 printDeviceName();
	 devices.resize(resources.getSize());

	 DEBUG("Initializing GPUs!");
	 initGPUs();
	 // TODO: init pedestal maps
	 currentBlock.reserve(GPU_FRAMES);
	 DEBUG("elements in the ringbuffer: " << resources.getNumberOfElements());
	 DEBUG("End of constructor!");
 }

 Uploader::~Uploader() { freeGPUs(); }

void Uploader::printDeviceName() {
	struct cudaDeviceProp prop;
	int numDevices;

	HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
	for(int i = 0; i < numDevices; ++i) {
		HANDLE_CUDA_ERROR(cudaSetDevice(i));
		HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
		std::cout << "Device #" << i << ":\t" << prop.name << std::endl;
	}
}

bool Uploader::upload(std::vector<Datamap>& data)
{
	//TODO: waht to do with a small amount of frames when terminating?
	for (std::size_t i = 0; i < data.size(); ++i) {
		if (currentBlock.size() == GPU_FRAMES) {
			if (!calcFrames(currentBlock)) {

				//TODO: find a better solution below
				//remove all used frames from the front
				for(std::size_t j = data.size() - i; j > 0; --j) {
					data[j-1] = data[i+j-1];
				}

				for(std::size_t j = 0; j < i; ++j)
					data.pop_back();

				return false;
			}

			currentBlock.clear();
		}
		currentBlock.push_back(data[i]);
	}

	data.clear();
	return true;
}

 std::vector<Photonmap> Uploader::download()
 {
	 std::vector<Photonmap> ret;
	 int current = nextFree;
	 std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
	 struct deviceData* dev = &Uploader::devices[current];

	 if(devices[nextFree].state != READY)
		 return ret;
	 nextFree = (nextFree + 1) % resources.getSize();

	 for (size_t i = 0; i < numPhotons; i += dimX * dimY)
		 ret.emplace_back(dimX, dimY, &dev->photon_host_raw[i]);

	 dev->photon_host.clear();
	 DEBUG("setting " << current << " to FREE");
	 dev->state = FREE;

	 if(!resources.push(&devices[current])) {
		 fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
		 exit(EXIT_FAILURE);
	 }
   
	 //TODO: remove debug below 
	 int all_empty = 1;
	 for(std::size_t i = 0; i < ret.size(); ++i)
		 if(!isMapEmpty(ret[i], dimX, dimY))
			 all_empty = 0;
	 /*		 else
			 DEBUG("map " << i << " is empty");
	 */
	 DEBUG("maps empty? " << (all_empty ? "yes" : "no"));

	 DEBUG("resources in use: " << resources.getNumberOfElements());
	 return ret;
 }

 void Uploader::initGPUs()
 {
	 DEBUG("initGPU()");

	 //TODO: init pedestalmaps!
	 for (std::size_t i = 0; i < devices.size(); ++i) {

		 devices[i].gain_host = &gain;
		 devices[i].pedestal_host = &pedestal;

		 DEBUG("setting " << i << " to FREE");
		 devices[i].state = FREE;
		 //TODO: is this really needed? if yes, throw out device member
		 devices[i].id = i;
		 devices[i].device = i / STREAMS_PER_GPU;

		 HANDLE_CUDA_ERROR(cudaSetDevice(devices[i].device));

		 HANDLE_CUDA_ERROR(cudaStreamCreate(&devices[i].str));

		 DEBUG("Allocating GPU memory on device for #" << i);
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].gain, dimX * dimY * sizeof(double) * 3));
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].pedestal, dimX * dimY * sizeof(uint16_t) * 3));
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].data, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].photons, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));

		 HANDLE_CUDA_ERROR(cudaMallocHost((void**)&devices[i].data_pinned, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));
		 HANDLE_CUDA_ERROR(cudaMallocHost((void**)&devices[i].photon_pinned, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));

		 uploadGainmap(devices[i]);
		 uploadPedestalmap(devices[i]);

		 synchronize();

		 if (!resources.push(&devices[i])) {
			 fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
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
		 HANDLE_CUDA_ERROR(cudaFreeHost(devices[i].photon_pinned));
		 HANDLE_CUDA_ERROR(cudaFreeHost(devices[i].data_pinned));
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
	 HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	 HANDLE_CUDA_ERROR(cudaMemcpy(stream.gain, stream.gain_host->at(0).data(), stream.gain_host->at(0).getSizeBytes() * 3, cudaMemcpyHostToDevice));
 }

 void Uploader::uploadPedestalmap(struct deviceData stream)
 {
	 HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	 HANDLE_CUDA_ERROR(cudaMemcpy(stream.pedestal, stream.pedestal_host->at(0).data(), stream.pedestal_host->at(0).getSizeBytes() * 3, cudaMemcpyHostToDevice));
 }

 void Uploader::downloadGainmap(struct deviceData stream)
 {
	 HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	 HANDLE_CUDA_ERROR(cudaMemcpy(stream.gain_host->at(0).data(), stream.gain, stream.gain_host->at(0).getSizeBytes() * 3, cudaMemcpyDeviceToHost));
 }

 void Uploader::downloadPedestalmap(struct deviceData stream)
 {
	 HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
	 HANDLE_CUDA_ERROR(cudaMemcpy(stream.pedestal_host->at(0).data(), stream.pedestal, stream.pedestal_host->at(0).getSizeBytes() * 3, cudaMemcpyDeviceToHost));
 }

 bool Uploader::calcFrames(std::vector<Datamap>& data)
 {
	 std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
	 std::vector<Photonmap> photonMaps;
	 photonMaps.reserve(GPU_FRAMES);

	 if(data.empty())
		 return false;

	 struct deviceData* dev;
	 if(!resources.pop(dev))
		 return false;

    dev->photon_host_raw = (uint16_t*)malloc(numPhotons * sizeof(uint16_t));
    if (!dev->photon_host_raw) {
        fputs("FATAL ERROR (Memory): Allocation failed!\n", stderr);
        exit(EXIT_FAILURE);
    }

	HANDLE_CUDA_ERROR(cudaSetDevice(dev->device));

	 DEBUG("copyin to pinned memory");
	 HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->data_pinned, data[0].data(), dimX * dimY * GPU_FRAMES, cudaMemcpyHostToHost, dev->str));

	 DEBUG("setting " << dev->id << " to PROCESSING");
	 dev->state = PROCESSING;
	 uploadToGPU(*dev, data);

	 calculate<<<dimX, dimY, 3 * (sizeof(uint16_t) + sizeof(double)) * dimY, dev->str>>>(uint16_t(dimX * dimY), dev->pedestal, dev->gain, dev->data, uint16_t(GPU_FRAMES), dev->photons);
     CHECK_CUDA_KERNEL;

	 downloadFromGPU(*dev);
	 
	 DEBUG("copying data from gpu to pinned memory");
	 DEBUG(dev->photon_host_raw << " <- " << dev->photon_pinned);
	 HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->photon_host_raw, dev->photon_pinned, dimX * dimY * GPU_FRAMES, cudaMemcpyHostToHost, dev->str));

	 DEBUG("Creating callback ...");
	 HANDLE_CUDA_ERROR(cudaStreamAddCallback(dev->str, Uploader::callback, &dev->id, 0));

	 return true;
 }

 void CUDART_CB Uploader::callback(cudaStream_t stream, cudaError_t status, void* data) {
	 //suppress "unused variable" compiler warning
	 (void)stream;

	 if(data == NULL) {
		 fputs("FATAL ERROR (callback): Missing index!\n", stderr);
		 exit(EXIT_FAILURE);
	 }

	 //TODO: move HANDLE_CUDA_ERROR out of the callback function
	 HANDLE_CUDA_ERROR(status);

	 Uploader::devices[*((int*)data)].state = READY;
 }

void Uploader::uploadToGPU(struct deviceData& dev, std::vector<Datamap>& data)
{
    std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
	std::size_t copySize = numPhotons * sizeof(*dev.data);

	if(data.empty())
		return;

    HANDLE_CUDA_ERROR(cudaSetDevice(dev.device));
	DEBUG("upload size: " << copySize);

	HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev.data, dev.data_pinned, copySize, cudaMemcpyHostToDevice, dev.str));
}

void Uploader::downloadFromGPU(struct deviceData& dev)
{
    std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
	std::size_t copySize = numPhotons * sizeof(*dev.photons);

	//TODO: copy back photons
    HANDLE_CUDA_ERROR(cudaSetDevice(dev.device));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev.photon_pinned, dev.photons, copySize, cudaMemcpyDeviceToHost, dev.str));
}

