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

The kernel is executed for each pixel individually and concurrently. In addition to this the kernel iterates over the same pixel of multiple frames sequentually to reduce overhead. Currently, there are 3 different kernels in use: one for the pedestal calibration, one for the calculation of the data and one to sum up a certain number of frames. 
The calibration kernel calculates the RMS from 2999 of the 3 gain stages. This is then used in the data calculation kernel where it is updated if dark values are detected. The last kernel, the summation kernel, sums a certein number of frames up. 

## Pedestal calibration

The pedestal calibration is executed in a similar fashion only that this time the uploadPedestal and calcPedestal functions are used. For the upload 2999 frames are required: 1000 for gain stage 0, 1000 for gain stage 1 and 999 for gain stage 2.

## Data calculation kernel

The calculation kernel applies corrections to the raw data which are the substraction of the pedestal values and the multiplication with the gain values according to the current gain stage. After this is done, the pedestal values are updated if dark pixels are detected. For this a weighted moving average is used. 
