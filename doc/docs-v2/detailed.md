# Features

### Data Transformation
- [Energy Conversion](#Energy-Conversion)
- Photon Counting

### Data Selection
- Pixel Masking

### Data Compression
- Clustering
- Frame Summation
- Maximum Value

# Energy Conversion
The energy conversion algorithm transforms the raw output data of a detector module into 2D matrices of calibrated energy values. This calculation step is a prerequisite for most use-cases, as most algorithms operate on energy maps. **Note:** This algorithm expects reciprocal gain maps. If needed, reciprocals can be calculated during the program initialization phase using the `GainMapInversion` kernel.
![GitHub Logo](img/energy_conversion.svg)

#### Inputs
- array of detector module output data
- array of reciprocal gain maps (one per gain stage)
- array of pedestal maps (one per gain stage)
- number of frames to be processed
- **optional:** pixel mask (2D array of boolean values: `true` = pixel is part of the calculation, `false` = pixel is ignored)

#### Outputs
- energy maps (array of 2D matrices containing calibrated energy values)
- gain stage maps (array of 2D matrices containing the gain stage of each pixel in the current frame)
