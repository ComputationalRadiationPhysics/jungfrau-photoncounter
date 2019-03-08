# Description

This file aims to provide some insight into the workflow of the photon counter. 

## Data entry

Once data enters the system via the uploadData function form the Dispencer class, the Dispencer will try to push this data into a ringbuffer and let's the user know how much frames have been pushed to this buffer. 

## Ringbuffer

The ringbuffer contains objects of the DeviceData struct which contains GPU memory for pedestal, gain, sum and data maps; an id, the number of maps, a state; host device, stream and event pointer as well as CPU memory for data and sum maps. 
For optimal distribution, the buffer has 3 states: FREE, PROCESSING and ready. At first all elements are marked as FREE. As soon as elements are uploaded, the corrosponding cells are marked as PROCESSING. When the processing is done the cell is marked as READY. 

![Hardware Overview](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/master/doc/hardware_diagram.png)

## Processing

Once data is pushed into the ringbuffer and the cell is marked as processing, the actual processing begins. For this, the next free GPU is selected and the data is first copied to the GPU and the last state of the pedestal maps is copied from the last processing GPU to the current GPU. Afterwards, the kernel functions are executed. The state of this device is than set to READY. 
Now it is possible to download the data using the downloadData function, where the data is copied back from the GPUs to the host memory and returned to the user. If this is not possible, the function returns false. 

![Algorithm Overview](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/master/doc/flowchart.png)

## Kernel

The kernel is executed for each pixel individually and concurrently. In addition to this the kernel iterates over the same pixel of multiple frames sequentually to reduce overhead. Currently, there are 10 different kernels in use:
* pedestal calibration
* energy calculation
* photon calculation
* clustering
* drift calculation
* gain stage masking (for debug output)
* gain map inversion
* reduction (to find maximal values)
* gathering maximal values from frames
* one to sum up a certain number of (energy) frames

The calibration kernel calculates the RMS from 2999 of the 3 gain stages. This is then used in the data calculation kernels (energy calculation, photon calculation and clustering) where it is updated if dark values are detected. The summation kernel, sums a certein number of frames up. The drift calculation tracks the pedestal drift over time. The gain stage masking kernel creates a gain stage map which can than be downloaded and viewed (mostly useful for debugging purposes). The gain stage inversion kernel inverts the gain stages (once at the start of the program) which speeds up data processing since multiplication can be done faster than division. The reduction kernel along with the gathering of the maximal values fill an array with the maximal calculated energy value for each frame (which is also used mostly for debugging purposes).

## Pedestal calibration

The pedestal [calibration kernel](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/clustering/include/jungfrau-photoncounter/kernel/Calibration.hpp) initially adds up ```N``` (user selectible at compile time) dark values in the gain stage 0 (which contains 1000 images). Every further dark value updates the rolling window (i.e. the mean is substracted and the new value is added). This results in the average over the roling window.
For the other gain stages a normal average over all calibration images (1000 for gain stage 1; 999 for gain stage 2) is used.

## Data calculation kernels

The calculation kernels, and in particular the [energy calculation kernel](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/clustering/include/jungfrau-photoncounter/kernel/Conversion.hpp), applies corrections to the raw data which are the substraction of the pedestal values and the multiplication with the gain values according to the current gain stage. After this is done, the pedestal values are updated if dark pixels are detected. For this a weighted moving average is used. 

The [photon calculation kernel](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/clustering/include/jungfrau-photoncounter/kernel/PhotonFinder.hpp) additionally converts the values to the number of photons. 

The [clustering kernel](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/clustering/include/jungfrau-photoncounter/kernel/ClusterFinder.hpp) extracts clusters around photon hits. A cluster is always centered around the largest element in the cluster. In addition to this, the cluster mustn't exceed the image dimensions. If these conditions are met, the cluster (i.e. the frame number, the position and the cluster values) is written into a cluster array where it is downloaded onto the host later on. Since the cluster condition of one pixel requires the neighboring pixels to be in the same calculation stage, the device/GPU needs to be completely synchronized which is currently only achievable by letting the kernel process only one frame at a time and finish afterwards. 
