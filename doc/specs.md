## Requirements

### General

(for general information on the process see [Jungfrau_GPU.pdf](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/master/doc/Jungfrau_GPU.pdf) and the presentations [2017_04_08](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/master/doc/presentation_2017_04_08/psi_presentation.pdf) and [2017_01_31](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/master/doc/presentation_2017_01_31/jungfrau-photoncounter_eng.pdf))

- process up to 32 modules (this requires handling throughput around 10 GB/s and mutliple modules per node)
- possibility to process different sized modules (mainly Jungfrau and Mönch)
- flag to switch between photon and energy output
- sum every 10-100 images
- possibility to mask single pixels (manually and automatically)
- initial pedestal calibration
- continuous online correction of the pedestal data
- cluster detection and image sharpening using this technique
- live displaying capabilities

### Pedestal callibration

- take ~1000 dark frames for each gain stage (not necessarily successive)
- calculate RMS
- only gain stage 0 pedestal values updatable since the other stages are only switched to when an actual signal is detected

### Online pedestal correction

- update pedestal values with every frame using its dark pixels
- use moving average as described by [John D. Cook](https://www.johndcook.com/blog/standard_deviation/)
- log difference between initial calibration and the current state of the pedestal maps
- resetable by user (stop online correction afterwards and fall back to initial callibration data)
- if a recallibration is needed, restart the program

### Pixel Mask

- a pixel mask map is linked to every JF module, in the same way that each JF module has 3 pedestal maps and 3 gains maps
- the map is composed of 0 (=mask) and 1 (= do not mask)
- before calculating stuff for each pixel, check it isn't masked.
- to create the first version of the mask, use the pedestal data files. Check that during the first 1000 frames, the gain bit of every pixel in every frame is 0. The next 1000 frames the gain bit of every pixel in every frame should be 1, the final 1000 frames the gain bit of every pixel in every frame should be 3. If this is ever not the case, mask the pixel.
- later: to update the pixel mask you could think about masking pixels which have a large pedestal uncertainty
- possible feature: sometimes we know we want to mask a whole chip before even looking at the data. Could this be an input parameter?
- output: the calculated pixel mask map should probably be saved as an image (same as for the pedestal maps) just to check it looks sensible.

### Live System

- 1-5 Hz
- show photon maps, current pedestal map, gain stage map and pedestal drift data
- show highest value
- displayable on different computer
- only show 'interesting' frames (frames with a certain amount pixels which are not in gain stage 0)

### Clustering

- optimize this for GPUs
- make window size programmable
- consists of clustering and interpolation (see [Cartier_2015_J._Inst._10_C03022.pdf](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/files/1521193/Cartier_2015_J._Inst._10_C03022.pdf))
- possible size are 3x3 (Mönch), 2x2 (Jungfrau), 5x5, 7x7, 9x9 and 11x11
