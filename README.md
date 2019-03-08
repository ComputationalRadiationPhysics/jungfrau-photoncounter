# jungfrau-photoncounter
Conversion of Jungfrau pixel detector data to photon count rate. 

# Installing

What is needed?
- alpaka (see [alpaka](https://github.com/ComputationalRadiationPhysics/alpaka) or install through this [spack repo](https://github.com/ComputationalRadiationPhysics/spack-repo) using the [spack package manager](https://spack.io/))
- boost ([dependency of alpaka](https://github.com/ComputationalRadiationPhysics/alpaka#dependencies))
- CUDA 8.0 (preferably 9.2 or higher; only needed if GPU are being used)
- CMake 3.11 or newer
- a compatible compiler (see [alpaka](https://github.com/ComputationalRadiationPhysics/alpaka))

See also [Issue #42](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/issues/42#issuecomment-457198329).

After downloading the repository, make sure that all the data files are in palce. The pedestal cailibration data (1000 images for stage G0, 1000 images for stage G1, 999 images for stage G2) has to be located at `jungfrau-photoncounter/data_pool/px_101016/allpede_250us_1243__B_000000.dat`, the gain maps (one for every stage) need to be located at `jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin` and the image data has to be located at `jungfrau-photoncounter/data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat`. For more information see [doc/usage.md](doc/usage.md).

To configure this project use:
```
mkdir build
cd build
ccmake .. #alternatively: cmake .. -DALPAKA_CUDA_ARCH=60 -DCMAKE_BUILD_TYPE=Release
```
**Important note: The debug mode is currently not working due to a bug. The release mode however, should work without problems. This will be fixed soon!**
Note: After selecting the desired backend, the configuration can be completed with the keys `c` (for configure) and `g`(for generate).
 
Now the compilation process can be started with:
```
make
```

# Documentation

Most parts of the documentation can be found in the [doc](doc) folder. 

The [description.md](doc/description.md) file contains a general overview of the workflow. The usage of the interface can be found in the [usage.md](doc/usage.md) file. 

The required specs for this project can be found inside the [specs.md](doc/specs.md) file. 

Informations about the sensor are located in the [Jungfrau_GPU.pdf](doc/Jungfrau_GPU.pdf) file. 

The slides of previous presentations for the project can be found in the [presentation_2017_01_31](doc/presentation_2017_01_31/jungfrau-photoncounter_eng.pdf) and [presentation_2017_04_08](doc/presentation_2017_04_08/psi_presentation.pdf) files. 

The source code documantation can be generated using doxygen or read directly in the source code. 

# License

This project is licensed under the GNU General Public License version 3. For more information see [LICENSE](https://github.com/ComputationalRadiationPhysics/jungfrau-photoncounter/blob/master/LICENSE).
