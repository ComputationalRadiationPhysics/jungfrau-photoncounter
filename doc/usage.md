**Please note that this software is still in develompment and the interface may change at any time!**

## Current main.cpp

The pedestal cailibration data (1000 images for stage G0, 1000 images for stage G1, 999 images for stage G2) has to be located at `jungfrau-photoncounter/data_pool/px_101016/allpede_250us_1243__B_000000.dat`, the gain maps (one for every stage) need to be located at `jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin` and the image data has to be located at`jungfrau-photoncounter/data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat`.
If other locations should be used than the `main.cpp` code has to be modified (see below).
The current `main.cpp` file is configured to work with CUDA devices. If it is preferred otherwise, the file has to be modified (see below).

## Usage

First, the data should be loaded. For this wi provide the class `Filecache` but this is not strictly required. It is also possible to provide other data sources, as long as the memory is pinned using alpaka.
```
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

After this step, the `Dispenser` class can be initialized:
```
Dispenser<Accelerator> dispenser(gain);

// calibrate pedestal data
dispenser.uploadPedestaldata(pedestaldata);
```

Now it is possible to upload (`std::size_t dispenser.uploadData(data, offset)`) and download (`bool dispenser.downloadData(&photon, &sum)`) data. It is recommended process all data with a loop:
```
Maps<Photon, Accelerator> photon{};
Maps<PhotonSum, Accelerator> sum{};
std::size_t offset = 0;
std::size_t downloaded = 0;

while (downloaded < data.numMaps) {
  offset = dispenser.uploadData(data, offset);
  if (dispenser.downloadData(&photon, &sum))
    downloaded += DEV_FRAMES;
}
```

The upload function returns the number of uploaded frames and the download function returns true if frames could be downloaded.

A current example can be found in the [main.cpp](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/master/src/main.cpp).
