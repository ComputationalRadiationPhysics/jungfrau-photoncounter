#include "Upload.hpp"

std::size_t Uploader::nextFree = 0;
std::vector<deviceData> Uploader::devices;

template <typename T>
T* allocateFrames(bool header, bool host, std::size_t n)
{
    T* ret;
	std::size_t size = DIMX * DIMY * sizeof(T) * n;
	
	if(header)
		size += n * FRAME_HEADER_SIZE;
			
    if (host)
        HANDLE_CUDA_ERROR(
            cudaMallocHost((void**)&ret, size));
    else
        HANDLE_CUDA_ERROR(
            cudaMalloc((void**)&ret, size));
    return ret;
}

Uploader::Uploader(Gainmap gain, Pedestalmap pedestal,
                   std::size_t numberOfDevices)
    : gain(gain), pedestal(pedestal),
      resources(STREAMS_PER_GPU * numberOfDevices)
{
    // check pedestal and gain map size
    if (gain.getN() != 3 || pedestal.getN() != 3) {
        char errorString[1000];
        snprintf(errorString, 1000, "FATAL ERROR (Map loading): %zu Pedestal "
                                    "maps and %zu Gain maps loaded! Exactly 3 "
                                    "of each are needed!\n",
                 pedestal.getN(), gain.getN());
        fputs(errorString, stderr);
        exit(EXIT_FAILURE);
    }

    // init remaining vars
    printDeviceName();
    devices.resize(resources.getSize());
    initGPUs();
    // TODO: init pedestal maps
    DEBUG("End of constructor!");
}

Uploader::~Uploader() { freeGPUs(); }

void Uploader::printDeviceName() const
{
    struct cudaDeviceProp prop;
    int numDevices;

    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
    for (int i = 0; i < numDevices; ++i) {
        HANDLE_CUDA_ERROR(cudaSetDevice(i));
        HANDLE_CUDA_ERROR(cudaGetDeviceProperties(&prop, i));
        std::cout << "Device #" << i << ":\t" << prop.name << std::endl;
    }
}

bool Uploader::isEmpty() const {
	return resources.isFull();
}

std::size_t Uploader::upload(Datamap& data, std::size_t offset)
{
	std::size_t ret = offset;
	
    // upload and process data package efficiently
    for (std::size_t i = ret; i < data.getN() + GPU_FRAMES; i += GPU_FRAMES) {
        Datamap current(GPU_FRAMES, data.data() + i);
        if (!calcFrames(current))
            return ret;
        ret += GPU_FRAMES;
    }

    // flush the remaining data
    if (ret < data.getN()) {
        Datamap current(GPU_FRAMES, data.data() + ret);
        ret += calcFrames(current);
    }

    return ret;
}

Photonmap Uploader::download()
{
    int current = nextFree;
    struct deviceData* dev = &Uploader::devices[current];

    if (devices[nextFree].state != READY)
        return Datamap(0, NULL);
    nextFree = (nextFree + 1) % resources.getSize();
	
    std::size_t num_frames = dev->num_frames;
    Datamap ret(num_frames, dev->photon_host);

    dev->state = FREE;

    if (!resources.push(&devices[current])) {
        fputs("FATAL ERROR (RingBuffer): Unexpected size!\n", stderr);
        exit(EXIT_FAILURE);
    }

    DEBUG("resources in use: " << resources.getNumberOfElements());
    return ret;
}

void Uploader::initGPUs()
{
    DEBUG("initGPU()");

    // TODO: init pedestalmaps!
    for (std::size_t i = 0; i < devices.size(); ++i) {

        devices[i].num_frames = 0;

        devices[i].gain_host = &gain;
        devices[i].pedestal_host = &pedestal;

        devices[i].state = FREE;
		devices[i].id = i;
        devices[i].device = i / STREAMS_PER_GPU;

        HANDLE_CUDA_ERROR(cudaSetDevice(devices[i].device));
        HANDLE_CUDA_ERROR(cudaStreamCreate(&devices[i].str));

        devices[i].gain = allocateFrames<GainType>(false, false, 3);
        devices[i].pedestal = allocateFrames<PedestalType>(false, false, 3);
        devices[i].data = allocateFrames<DataType>(true, false, GPU_FRAMES);
        devices[i].photon = allocateFrames<PhotonType>(true, false, GPU_FRAMES);
        devices[i].data_pinned = allocateFrames<DataType>(true, true, GPU_FRAMES);
        devices[i].photon_pinned = allocateFrames<PhotonType>(true, true, GPU_FRAMES);

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
        HANDLE_CUDA_ERROR(cudaFree(devices[i].photon));
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
    HANDLE_CUDA_ERROR(cudaMemcpy(stream.gain, stream.gain_host->data(),
                                 stream.gain_host->getSizeBytes(),
                                 cudaMemcpyHostToDevice));
}

void Uploader::uploadPedestalmap(struct deviceData stream)
{
    HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
    HANDLE_CUDA_ERROR(cudaMemcpy(stream.pedestal, stream.pedestal_host->data(),
                                 stream.pedestal_host->getSizeBytes(),
                                 cudaMemcpyHostToDevice));
}

void Uploader::downloadGainmap(struct deviceData stream)
{
    HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
    HANDLE_CUDA_ERROR(cudaMemcpy(stream.gain_host->data(), stream.gain,
                                 stream.gain_host->getSizeBytes(),
                                 cudaMemcpyDeviceToHost));
}

void Uploader::downloadPedestalmap(struct deviceData stream)
{
    HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
    HANDLE_CUDA_ERROR(cudaMemcpy(stream.pedestal_host->data(), stream.pedestal,
                                 stream.pedestal_host->getSizeBytes(),
                                 cudaMemcpyDeviceToHost));
}

int Uploader::calcFrames(Datamap& data)
{
	// load available device and number of frames
    std::size_t num_photons = DIMX * DIMY * data.getN();
    struct deviceData* dev;
    if (!resources.pop(dev))
        return false;
    dev->num_frames = data.getN();

	// allocate memory for the photon data
    dev->photon_host = (PhotonType*)malloc(num_photons * sizeof(PhotonType));
    if (!dev->photon_host) {
        fputs("FATAL ERROR (Memory): Allocation failed!\n", stderr);
        exit(EXIT_FAILURE);
    }
	
	// set state to processing
    dev->state = PROCESSING;

	// select device
    HANDLE_CUDA_ERROR(cudaSetDevice(dev->device));

	/*	
//TODO: remove this???
	//theoretically not needed because the data is loaded into pinned memory
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->data_pinned, data.data(),
                                      num_photons * sizeof(DataType),
                                      cudaMemcpyHostToHost, dev->str));
*/
	//upload data to GPU
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->data, data.data(),//dev->data_pinned,
                                      num_photons * sizeof(DataType),
                                      cudaMemcpyHostToDevice, dev->str));

	//TODO: optimize kernel vars???
	//calculate photon data and check for kernel errors
    calculate<<<DIMX, DIMY, 6 * sizeof(GainType) * DIMY, dev->str>>>(
        DIMX * DIMY, dev->pedestal, dev->gain, dev->data, dev->num_frames,
        dev->photon);
    CHECK_CUDA_KERNEL;

	//download data from GPU to pinned memory
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->photon_pinned, dev->photon,
                                      num_photons * sizeof(PhotonType),
                                      cudaMemcpyDeviceToHost, dev->str));

	//copy data to 'less expensive' memory
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->photon_host, dev->photon_pinned,
                                      num_photons * sizeof(PhotonType),
                                      cudaMemcpyHostToHost, dev->str));

	//create callback function
    DEBUG("Creating callback ...");
    HANDLE_CUDA_ERROR(
        cudaStreamAddCallback(dev->str, Uploader::callback, &dev->id, 0));

	//return number of processed frames
    return GPU_FRAMES;
}

void CUDART_CB Uploader::callback(cudaStream_t stream, cudaError_t status,
                                  void* data)
{
	// suppress "unused variable" compiler warning
    (void)stream;

    if (data == NULL) {
        fputs("FATAL ERROR (callback): Missing index!\n", stderr);
        exit(EXIT_FAILURE);
    }

	DEBUG(*((int*)data) << " is now ready to process!");

    // TODO: move HANDLE_CUDA_ERROR out of the callback function
    HANDLE_CUDA_ERROR(status);

    Uploader::devices[*((int*)data)].state = READY;
}
