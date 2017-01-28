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
    : gain(gain), pedestal(pedestal),/* dimX(dimX), dimY(dimY),*/
      resources(STREAMS_PER_GPU * numberOfDevices)
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
	DEBUG("uploading " << data.size() << "maps");
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

				DEBUG("new size at " << i << " = " << data.size());
				return false;
			}

			currentBlock.clear();
		}
		currentBlock.push_back(data[i]);
	}

	DEBUG("getting out! Resources available: " << resources.getNumberOfElements());
	data.clear();
	return true;
}

 std::vector<Photonmap> Uploader::download()
 {
	 std::vector<Photonmap> ret;
	 int current = nextFree;

	 if(devices[nextFree].state != READY)
		 return ret;
	 nextFree = (nextFree + 1) % resources.getSize();



	 std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
	 struct deviceData* dev = &Uploader::devices[current];

	 for (size_t i = 0; i < numPhotons; i += dimX * dimY) {
		 //TODO: use emplace back directly with ret
		 dev->photon_host.emplace_back(dimX, dimY, &dev->photon_host_raw[i]);
	 }

	 //TODO: remove debug below
	 /*for(std::size_t o = 0; o < GPU_FRAMES; ++o){
		 if(isMapEmpty(dev->photon_host[o], dimX, dimY))
			 DEBUG("map " << o << " is empty!");
			 }*/



	 ret = Uploader::devices[current].photon_host;
	 Uploader::devices[current].photon_host.clear();
	 DEBUG("setting " << current << " to FREE");
	 Uploader::devices[current].state = FREE;
	 if(!resources.push(&devices[current])) {
		 fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
		 exit(EXIT_FAILURE);
	 }
	 DEBUG("resources in use: " << resources.getNumberOfElements());
	 return ret;
 }

 void Uploader::initGPUs()
 {
	 DEBUG("initGPU()");

	 //TODO: init pedestalmaps!
	 for (std::size_t i = 0; i < devices.size(); ++i) {
		 DEBUG("Uploading Pedestalmaps for device " << i / STREAMS_PER_GPU << " with i=" << i);
		 devices[i].gain_host = &gain;

		 DEBUG("Uploading Gainmaps for device " << i / STREAMS_PER_GPU << " with i=" << i);
		 devices[i].pedestal_host = &pedestal;

		 DEBUG("setting " << i << " to FREE");
		 devices[i].state = FREE;
		 //TODO: is this really needed? if yes, throw out device member
		 devices[i].id = i;
		 devices[i].device = i / STREAMS_PER_GPU;

		 DEBUG("Setting device " << i / STREAMS_PER_GPU);
		 HANDLE_CUDA_ERROR(cudaSetDevice(i / STREAMS_PER_GPU));

		 DEBUG("Allocating GPU memory on device for #" << i);
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].gain, dimX * dimY * sizeof(double) * 3));
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].pedestal, dimX * dimY * sizeof(uint16_t) * 3));
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].data, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));
		 HANDLE_CUDA_ERROR(cudaMalloc((void**)&devices[i].photons, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));

		 HANDLE_CUDA_ERROR(cudaMallocHost((void**)&devices[i].data_pinned, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));
		 HANDLE_CUDA_ERROR(cudaMallocHost((void**)&devices[i].photon_pinned, dimX * dimY * sizeof(uint16_t) * GPU_FRAMES));

		 DEBUG("Creating GPU stream #" << i);
		 HANDLE_CUDA_ERROR(cudaStreamCreate(&devices[i].str));

		 synchronize();

		 DEBUG("Uploading Gainmaps for #" << i);
		 uploadGainmap(devices[i]);
		 DEBUG("Uploading Pedestalmaps for #" << i);
		 uploadPedestalmap(devices[i]);

		 DEBUG("elements in the ringbuffer: " << resources.getNumberOfElements());
		 DEBUG("is rb empty? " << resources.isEmpty());
		 DEBUG("is rb full? " << resources.isFull());

		 if (!resources.push(&devices[i])) {
			 fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
			 exit(EXIT_FAILURE);
		 }
	 }
	 DEBUG("number of elements in resources: " << resources.getNumberOfElements());
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
	 //DEBUG("calcFrames");
	 std::vector<Photonmap> photonMaps;
	 photonMaps.reserve(GPU_FRAMES);

	 if(data.empty()) {
		 DEBUG("no data .... doing nothing");
		 return false;
	 }

	 struct deviceData* dev;
	 if(!resources.pop(dev))
		 return false;


    std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
    dev->photon_host_raw = (uint16_t*)malloc(numPhotons * sizeof(uint16_t));
    if (!dev->photon_host_raw) {
        fputs("FATAL ERROR (Memory): Allocation failed!\n", stderr);
        exit(EXIT_FAILURE);
    }

	 DEBUG("copyin to pinned memory");
	 DEBUG("pinned data = " << dev->data_pinned << " & src = " << data[0].data());
	 HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->data_pinned, data[0].data(), dimX * dimY * GPU_FRAMES, cudaMemcpyHostToHost, dev->str));

	 DEBUG("Doing GPU stuff now");

	 DEBUG("setting " << dev->id << " to PROCESSING");
	 dev->state = PROCESSING;
	 uploadToGPU(*dev, data);

	 calculate<<<dimX, dimY, 3 * (sizeof(uint16_t) + sizeof(double)) * dimY, dev->str>>>(uint16_t(dimX * dimY), dev->pedestal, dev->gain, dev->data, uint16_t(GPU_FRAMES), dev->photons);
     CHECK_CUDA_KERNEL;

	 downloadFromGPU(*dev);
	 
	 DEBUG("copying data from gpu to pinned memory");
	 HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->photon_host_raw, dev->photon_pinned, dimX * dimY * GPU_FRAMES, cudaMemcpyHostToHost, dev->str));

	 DEBUG("Creating callback ...");
	 HANDLE_CUDA_ERROR(cudaStreamAddCallback(dev->str, Uploader::callback, &dev->id, 0));

	 return true;
 }

 void CUDART_CB Uploader::callback(cudaStream_t stream, cudaError_t status, void* data) {
	 //suppress "unused variable" compiler warning
	 (void)stream;

	 DEBUG("HELP ME I AM TRAPPED IN A SUPERCOMPUTER AND I CAN'T GET OUT!!!!");

	 if(data == NULL) {
		 fputs("FATAL ERROR (callback): Missing index!\n", stderr);
		 exit(EXIT_FAILURE);
	 }

	 HANDLE_CUDA_ERROR(status);
	 DEBUG("setting " << *((int*)data) << " to READY");

	 struct deviceData* dev = &Uploader::devices[*((int*)data)];
	 dev->state = READY;
	 DEBUG("stream: " << *((int*)data));
 }

void Uploader::uploadToGPU(struct deviceData& dev, std::vector<Datamap>& data)
{
	if(data.empty())
		return;
    HANDLE_CUDA_ERROR(cudaSetDevice(dev.device));
	DEBUG("upload size: " << data.size() * data[0].getSizeBytes());

	//TODO: clean up
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev.data, dev.data_pinned, data.size() * data[0].getSizeBytes(), cudaMemcpyHostToDevice, dev.str));
}

void Uploader::downloadFromGPU(struct deviceData& dev)
{
    std::size_t numPhotons = dimX * dimY * GPU_FRAMES;
	std::size_t copySize = numPhotons * sizeof(*dev.photons);

    HANDLE_CUDA_ERROR(cudaSetDevice(dev.device));
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev.photon_pinned, dev.photons, copySize, cudaMemcpyDeviceToHost, dev.str));
}

