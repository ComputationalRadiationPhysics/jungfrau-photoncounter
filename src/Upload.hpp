#pragma once

#include "Config.hpp"
#include "CudaHeader.hpp"
#include "Kernel.hpp"
#include "Pixelmap.hpp"
#include "RingBuffer.hpp"
#include <cmath>

enum ProcessingState { FREE, PROCESSING, READY };

struct deviceData {
    // IDs
    int device;
    int id;
    cudaStream_t str;
    // Pinned data pointer
    PhotonType* photon_host;
	PhotonType* photon_pinned;
	//GPU pointer
    PhotonType* photon;
    DataType* data;
	GainType* gain;
	PedestalType* pedestal;	
    // Maps
    Gainmap* gain_host;
    Pedestalmap* pedestal_host;
    // State
    ProcessingState state;
	//Number of frames
	std::size_t num_frames;
};

class Uploader {
public:
    Uploader(Gainmap gain, Pedestalmap pedestal, std::size_t numberOfDevices);
    Uploader(const Uploader& other) = delete;
    Uploader& operator=(const Uploader& other) = delete;
    ~Uploader();

	bool isEmpty() const;
	std::size_t upload(Datamap& data, std::size_t offset);
    Photonmap download();

    void synchronize();
    void printDeviceName() const;

protected:
private:
    RingBuffer<deviceData*> resources;
    static std::vector<deviceData> devices;
    static std::size_t nextFree;
    Gainmap gain;
    Pedestalmap pedestal;

    static void CUDART_CB callback(cudaStream_t stream, cudaError_t status,
                                   void* data);

    void initGPUs();
    void freeGPUs();

    void uploadGainmap(struct deviceData stream);
    void uploadPedestalmap(struct deviceData stream);

    void downloadGainmap(struct deviceData stream);
    void downloadPedestalmap(struct deviceData stream);

    int calcFrames(Datamap& data);
};
