# Usage

**Please note that this software is still in develompment and the interface may change at any time!**

## Current main.cpp

The pedestal cailibration data (1000 images for stage G0, 1000 images for stage G1, 999 images for stage G2) has to be located at `jungfrau-photoncounter/data_pool/px_101016/allpede_250us_1243__B_000000.dat`, the gain maps (one for every stage) need to be located at `jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin` and the image data has to be located at`jungfrau-photoncounter/data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat`.
If other locations should be used than the `main.cpp` code has to be modified (see below).
The current `main.cpp` file is configured to work with CUDA devices. If it is preferred otherwise, the file has to be modified (see below).

## Usage

First, the headers files need to be included and an alpaka Accelerator has to be selected:
```C++
#include "Alpakaconfig.hpp"
#include "Config.hpp"
#include "Dispenser.hpp"
#include "Filecache.hpp"

using Accelerator = GpuCudaRt;
```

The `Config.hpp` contains project specific configurations like the image size and the package size while the `Alpakaconfig.hpp` contains alpaka specific configurations. 

After this, the data should be loaded. For this we provide the class `Filecache` but this is not strictly required. It is also possible to provide other data sources, as long as the memory is pinned using alpaka.
```C++
// create a sufficiently large Filecache
Filecache fc(1024UL * 1024 * 1024 * 16);

// load pedestal data
Maps<Data, Accelerator> pedestaldata(
    fc.loadMaps<Data, Accelerator>("../../jungfrau-photoncounter/data_pool/"
      "px_101016/allpede_250us_1243__B_000000.dat",
    true));

// load gain maps
Maps<Gain, Accelerator> gain(fc.loadMaps<Gain, Accelerator>(
    "../../jungfrau-photoncounter/data_pool/px_101016/"
      "gainMaps_M022.bin"));

// load data
Maps<Data, Accelerator> data(
    fcloadMaps<Data, Accelerator>("../../jungfrau-photoncounter/data_pool/px_101016/"
      "Insu_6_tr_1_45d_250us__B_000000.dat",
    true));
```

This is now a good time to allocate space for the output data:
```C++
    // allocate space for the optional mask map
    FramePackage<MaskMap, Accelerator, Dim, Size> mask(SINGLEMAP);
    mask.numFrames = 0;
    
    // create empty, optional input mask
    boost::optional<typename Accelerator::HostBuf<MaskMap>>
        maskPtr;
    if (mask.numFrames == SINGLEMAP)
        maskPtr = mask.data;

    // allocate space for output data
    FramePackage<EnergyMap, Accelerator, Dim, Size> energy_data(DEV_FRAMES);
    FramePackage<PhotonMap, Accelerator, Dim, Size> photon_data(DEV_FRAMES);
    FramePackage<SumMap, Accelerator, Dim, Size> sum_data(DEV_FRAMES /
                                                                SUM_FRAMES);
    // allocate space for the clustering results if needed
    ClusterArray<Accelerator, Dim, Size> clusters_data(
                                         MAX_CLUSTER_NUM_USER * DEV_FRAMES);
    
    // allocate space for the maximal values
    FramePackage<EnergyValue, Accelerator, Dim, Size> maxValues_data(
        DEV_FRAMES);

    // create placeholder for output data
    boost::optional<FramePackage<EnergyMap, Accelerator, Dim, Size>&> energy =
        energy_data;
    boost::optional<FramePackage<PhotonMap, Accelerator, Dim, Size>&> photon =
        photon_data;
    boost::optional<FramePackage<SumMap, Accelerator, Dim, Size>&> sum =
        sum_data;
    
    // Note: remove the = cluster_data if the clustering result should not be
    //       downloaded
    boost::optional<ClusterArray<Accelerator, Dim, Size>&>
        clusters = clusters_data;
    boost::optional<FramePackage<EnergyValue, Accelerator, Dim, Size>&>
        maxValues = maxValues_data;
```

After this step, the `Dispenser` class can be initialized:
```C++
Dispenser<Accelerator> dispenser(gain, maskPtr);

// calibrate pedestal data
dispenser.uploadPedestaldata(pedestaldata);
```
For simultaneous of multiple modules, one instance of the `Dispenser` class per module is needed:
```C++
Dispenser<Accelerator> dispenser1(gain1, maskPtr1);
Dispenser<Accelerator> dispenser2(gain2, maskPtr2);
// ...

// calibrate pedestal data
dispenser1.uploadPedestaldata(pedestaldata1);
dispenser2.uploadPedestaldata(pedestaldata2);
//...
```

Now it is possible to upload (`std::size_t uploadData(data, offset, flags)`) and download (`std::size_t downloadData(energy, photon, sum, maxValues, clusters)`) data. It is recommended process all data with a loop:
```C++
    std::size_t offset = 0;
    std::size_t downloaded = 0;
    std::size_t currently_downloaded_frames = 0;

    ExecutionFlags ef;
    ef.mode = 1; // photon and energy values are calculated
    ef.summation = 1;
    ef.masking = 1;
    ef.maxValue = 1;

    // process data maps
    while (downloaded < data.numFrames) {
        offset = dispenser->uploadData(data, offset, ef);
        if (currently_downloaded_frames = dispenser->downloadData(
                energy, photon, sum, maxValues, clusters)) {
            downloaded += currently_downloaded_frames;

            DEBUG(downloaded << "/" << data.numFrames << " downloaded; "
                             << offset << " uploaded");
        }
    }
```

The upload function returns the number of uploaded frames and the download function returns true if frames could be downloaded.

A current example can be found in the [main.cpp](../src/main.cpp).
