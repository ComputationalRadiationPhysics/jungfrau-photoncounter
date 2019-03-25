# Features

### Data Transformation
- [Energy Conversion](#Energy-Conversion)
- [Photon Finder](#Photon-Finder)
- [Gain Map Inversion](#Gain-Map-Inversion)

### Data Selection
- Pixel Masking

### Data Compression
- Clustering
- Frame Summation
- Maximum Value

# Energy Conversion
The energy conversion algorithm transforms the raw output data of a detector module into 2D matrices of calibrated energy values. This calculation step is a prerequisite for most use-cases, as most algorithms operate on energy maps. **Note:** This algorithm expects reciprocal gain maps. If needed, reciprocals can be calculated during the program initialization phase using the `GainMapInversion` kernel.

<p align="center">
  <img alt="Figure: Energy Conversion" src="img/energy_conversion.svg" />
</p>

#### Inputs
- array of detector module output data
- array of reciprocal gain maps (one per gain stage)
- array of pedestal maps (one per gain stage)
- number of frames to be processed
- **optional:** pixel mask (2D array of boolean values: `true` = pixel is part of the calculation, `false` = pixel is ignored)

#### Outputs
- energy maps (array of 2D matrices containing calibrated energy values)
- gain stage maps (array of 2D matrices containing the gain stage of each pixel in the current frame)

#### Usage
The algorithm is executed as an Alpaka kernel. Inputs are passed as pointers which must be accessible from the chosen Alpaka accelerator. Optional parameters accept `nullptr`. 

**TODO: code example**

# Photon Finder
The Photon Finder algorithm converts raw output data of a detector module to photon maps, using pedestal maps and reciprocal gain maps. **Note:** If reciprocal gain maps are not available, they can be inverted using the [Gain Map Inversion](#Gain-Map-Inversion) kernel.

# Gain Map Inversion
The Energy Conversion algorithm requires reciprocal gain maps. If reciprocals are not immediately loaded, this kernel can be used to invert an array of gain maps to their reciprocals. This step is usually executed once in the program initialization phase when the Alpaka accelerators are provided with the required values for data processing.

<p align="center">
  <img alt="Figure: Gain Map Inversion" src="img/gain_map_inversion.svg" width="500em"/>
</p>

#### Inputs
- array of gain maps

#### Outputs
- array of reciprocal gain maps

#### Usage
The algorithm is executed as an Alpaka kernel. Inputs are passed as pointers which must be accessible from the chosen Alpaka accelerator. 

**TODO: code example**
