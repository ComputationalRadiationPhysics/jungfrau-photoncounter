#pragma once

#include "Kernel.hpp"
#include "Pixelmap.hpp"
#include "RingBuffer.hpp"
#include <cmath>

#define HANDLE_CUDA_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))
#define CHECK_CUDA_KERNEL (handleCudaError(cudaGetLastError(), __FILE__, __LINE__))

enum ProcessingState {
	FREE,
	PROCESSING,
	READY
};

//TODO: test different sizes
const std::size_t GPU_FRAMES = 1000;
const std::size_t STREAMS_PER_GPU = 4;

void handleCudaError(cudaError_t error, const char* file, int line);

struct deviceData {
    int device;
	int id;
    cudaStream_t str;
    // TODO: define types somewhere
    double* gain;
    uint16_t* pedestal;
    uint16_t* data;
    uint16_t* photons;
	uint16_t* photon_host_raw;
	uint16_t* photon_pinned;
	uint16_t* data_pinned;
    std::array<Gainmap, 3>* gain_host;
    std::array<Pedestalmap, 3>* pedestal_host;
	std::vector<Datamap> data_host;
	std::vector<Photonmap> photon_host;
	ProcessingState state;
};

class Uploader {
public:
	//TODO: use consistent names and fix types
	//TODO: add consts
    Uploader(std::array<Gainmap, 3> gain, std::array<Pedestalmap, 3> pedestal,
             std::size_t dimX, std::size_t dimY, std::size_t numberOfDevices);
    Uploader(const Uploader& other) = delete;
    Uploader& operator=(const Uploader& other) = delete;
    ~Uploader();

    bool upload(std::vector<Datamap>& data);
    std::vector<Photonmap> download();

    void synchronize();
	void printDeviceName();

protected:
private:
    RingBuffer<deviceData*> resources;
    static std::vector<deviceData> devices;
	static std::size_t nextFree;
    std::vector<Datamap> currentBlock;
    std::array<Gainmap, 3> gain;
    std::array<Pedestalmap, 3> pedestal;
    static std::size_t dimX, dimY;

	static void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void* data);

    void initGPUs();
    void freeGPUs();

    void uploadGainmap(struct deviceData stream);
    void uploadPedestalmap(struct deviceData stream);

    void downloadGainmap(struct deviceData stream);
    void downloadPedestalmap(struct deviceData stream);

    bool calcFrames(std::vector<Datamap>& data);
    void uploadToGPU(struct deviceData& dev, std::vector<Datamap>& data);
    void downloadFromGPU(struct deviceData& dev);
};
