#include "Upload.hpp"

std::size_t Uploader::nextFree = 0;
std::vector<deviceData> Uploader::devices;

template <typename T> T* allocateFrames(bool header, bool host, std::size_t n)
{
    T* ret;
    std::size_t size = DIMX * DIMY * sizeof(T) * n;

    if (header)
        size += n * FRAME_HEADER_SIZE;

    if (host)
        HANDLE_CUDA_ERROR(cudaMallocHost((void**)&ret, size));
    else
        HANDLE_CUDA_ERROR(cudaMalloc((void**)&ret, size));
    return ret;
}

Uploader::Uploader(Gainmap gain, std::size_t numberOfDevices)
    : gain(gain), resources(STREAMS_PER_GPU * numberOfDevices)
{
    // check gain map size
    if (gain.getN() != 3) {
        char errorString[1000];
        snprintf(errorString, 1000, "FATAL ERROR (Map loading): "
                                    "%zu Gain maps loaded! Exactly 3 "
                                    "are needed!\n",
                 gain.getN());
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

bool Uploader::isEmpty() const { return resources.isFull(); }

std::size_t Uploader::upload(Datamap& data, std::size_t offset)
{
    std::size_t ret = offset;
    std::size_t map_size =
        (DIMX * DIMY) * Datamap::elementSize + FRAME_HEADER_SIZE;

    // upload and process data package efficiently
    for (; ret <= data.getN() - GPU_FRAMES; ret += GPU_FRAMES) {
        DEBUG("-->current offset is "
              << ret * map_size << " (max: " << data.getSizeBytes() << ")");
        Datamap current(GPU_FRAMES, data.data() + ret * map_size / sizeof(DataType));

        // TODO: (below) is this even a valid state???
        if (!calcFrames(current))
            return ret;
    }

    // flush the remaining data
    if (ret < data.getN()) {
        DEBUG("-->last offset is " << ret * map_size
                                   << " (max: " << data.getSizeBytes() << ")");
        Datamap current(ret - data.getN(), data.data() + ret * map_size / sizeof(DataType));
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

    for (std::size_t i = 0; i < devices.size(); ++i) {

        devices[i].num_frames = 0;

        devices[i].gain_host = &gain;

        devices[i].state = FREE;
        devices[i].id = i;
        devices[i].device = i / STREAMS_PER_GPU;

        HANDLE_CUDA_ERROR(cudaSetDevice(devices[i].device));
        HANDLE_CUDA_ERROR(cudaStreamCreate(&devices[i].str));

        devices[i].gain = allocateFrames<GainType>(false, false, 3);
        devices[i].pedestal = allocateFrames<PedestalType>(false, false, 3);
        devices[i].data = allocateFrames<DataType>(true, false, GPU_FRAMES);
        devices[i].photon = allocateFrames<PhotonType>(true, false, GPU_FRAMES);
        devices[i].photon_pinned =
            allocateFrames<PhotonType>(true, true, GPU_FRAMES);

        uploadGainmap(devices[i]);

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

void Uploader::downloadGainmap(struct deviceData stream)
{
    HANDLE_CUDA_ERROR(cudaSetDevice(stream.device));
    HANDLE_CUDA_ERROR(cudaMemcpy(stream.gain_host->data(), stream.gain,
                                 stream.gain_host->getSizeBytes(),
                                 cudaMemcpyDeviceToHost));
}

int Uploader::calcFrames(Datamap& data)
{
    // load available device and number of frames
    std::size_t num_pixels = data.getPixelsPerFrame() * data.getN();
    struct deviceData* dev;
    if (!resources.pop(dev))
        return false;
    dev->num_frames = data.getN();

    // allocate memory for the photon data
    dev->photon_host = (PhotonType*)malloc(num_pixels * sizeof(PhotonType));
    if (!dev->photon_host) {
        fputs("FATAL ERROR (Memory): Allocation failed!\n", stderr);
        exit(EXIT_FAILURE);
    }

    // set state to processing
    dev->state = PROCESSING;

    // select device
    HANDLE_CUDA_ERROR(cudaSetDevice(dev->device));

	//upload data to GPU
	HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->data, data.data(),
                                  num_pixels * sizeof(DataType),
								  cudaMemcpyHostToDevice, dev->str));
 
    DEBUG("From: " << devices[(dev->id - 1) % devices.size()].device); 
    DEBUG("To: " << dev->device);
    // transfer pedestal data from last device
    HANDLE_CUDA_ERROR(cudaMemcpyPeerAsync(
        dev->pedestal,
        dev->device, 
        devices[(dev->id - 1) % devices.size()].pedestal,
        devices[(dev->id - 1) % devices.size()].device, 
        3 * DIMX * DIMY * sizeof(PedestalType), dev->str));


    // calculate photon data and check for kernel errors
    calculate<<<DIMX, DIMY, 0, dev->str>>>(DIMX * DIMY, dev->pedestal,
                                           dev->gain, dev->data,
                                           dev->num_frames, dev->photon);
    CHECK_CUDA_KERNEL;

    // download data from GPU to pinned memory
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->photon_pinned, dev->photon,
                                      num_pixels * sizeof(PhotonType),
                                      cudaMemcpyDeviceToHost, dev->str));

	//TODO: does this work this way, or do i have to copy the data in the callback function?????
    // copy data to 'less expensive' memory
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(dev->photon_host, dev->photon_pinned,
                                      num_pixels * sizeof(PhotonType),
                                      cudaMemcpyHostToHost, dev->str));

    // create callback function
    DEBUG("Creating callback ...");
    HANDLE_CUDA_ERROR(
        cudaStreamAddCallback(dev->str, Uploader::callback, &dev->id, 0));

    // return number of processed frames
    return GPU_FRAMES;
}

void Uploader::uploadPedestaldata(Datamap& pedestaldata)
{
    std::size_t offset = 0;
    std::size_t map_size =
        (DIMX * DIMY) * Datamap::elementSize + FRAME_HEADER_SIZE;

    // upload and process data package efficiently
    // TODO test; at least 2999 frames for pedestals
    for (; offset < pedestaldata.getN() - GPU_FRAMES;
         offset += GPU_FRAMES) {
        Datamap current(GPU_FRAMES, pedestaldata.data() + offset * map_size / sizeof(DataType));
        if (!calcPedestals(current, offset))
            return;

        DEBUG("Frames " << offset << " von " << (pedestaldata.getN() + GPU_FRAMES));
    }

    // flush the remaining data
    if (offset < pedestaldata.getN()) {
        Datamap current(GPU_FRAMES, pedestaldata.data() + offset * map_size / sizeof(DataType));
        calcPedestals(current, offset);
    }
}

int Uploader::calcPedestals(Datamap& pedestaldata, uint32_t num)
{
    // load available device and number of frames
    std::size_t num_pixels = pedestaldata.getPixelsPerFrame() * pedestaldata.getN();
    struct deviceData* dev;
    if (!resources.pop(dev))
        return false;
    dev->num_frames = pedestaldata.getN();

    // set state to processing
    dev->state = PROCESSING;

    // select device
    HANDLE_CUDA_ERROR(cudaSetDevice(dev->device));

    // upload data to GPU
    HANDLE_CUDA_ERROR(cudaMemcpy(
        dev->data, pedestaldata.data(),
        num_pixels * sizeof(DataType), cudaMemcpyHostToDevice));

    DEBUG("From: " << devices[(dev->id - 1) % devices.size()].device); 
    DEBUG("To: " << dev->device);

    // transfer pedestal data from last device
    HANDLE_CUDA_ERROR(cudaMemcpyPeer(
        dev->pedestal,
        dev->device, 
        devices[(dev->id - 1) % devices.size()].pedestal,
        devices[(dev->id - 1) % devices.size()].device, 
        pedestaldata.getSizeBytes()));

    // calculate photon data and check for kernel errors
    calibrate<<<DIMX, DIMY, 0, dev->str>>>(DIMX * DIMY, dev->num_frames, num,
                                           dev->data, dev->pedestal);
    CHECK_CUDA_KERNEL;

    dev->state = FREE;

    // return number of processed frames
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
