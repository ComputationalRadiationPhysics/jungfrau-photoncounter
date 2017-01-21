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

//TODO: find right size!
const std::size_t RINGBUFFER_SIZE = 1000;
//TODO: make dynamic & find right size
const std::size_t GPU_FRAMES = 2000;

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
    std::array<Gainmap, 3>* gain_host;
    std::array<Pedestalmap, 3>* pedestal_host;
	std::vector<Datamap> data_host;
	std::vector<Photonmap> photon_host;
	//TODO is the enum keyword really needed?
	enum ProcessingState state;
};

class Uploader {
public:
	//TODO: use consitent names and fix types
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
    std::vector<Datamap> currentBlock;
    // TODO: remove below (after all depencies are cleared)
    std::array<Gainmap, 3> gain;
    // TODO: remove below (after all depencies are cleared)
    std::array<Pedestalmap, 3> pedestal;
    std::size_t dimX, dimY;
    static std::vector<deviceData> devices;
	static std::size_t nextFree;

	static void CUDART_CB callback(cudaStream_t stream, cudaError_t status, void* data);

    void initGPUs();
    void freeGPUs();

    void uploadGainmap(struct deviceData stream);
    void uploadPedestalmap(struct deviceData stream);

    void downloadGainmap(struct deviceData stream);
    void downloadPedestalmap(struct deviceData stream);

    // OPTIONAL: implement memory counter to prevent too much data in memory
    // OPTIONAL: implement error handling

    bool calcFrames(std::vector<Datamap>& data);
    void uploadToGPU(struct deviceData& dev, std::vector<Datamap>& data);
    void downloadFromGPU(struct deviceData& dev);
};
