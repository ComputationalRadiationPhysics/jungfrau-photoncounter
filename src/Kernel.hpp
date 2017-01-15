#pragma once
#include <cstdint>
#include <math.h>

/**
 *  Kernel function with conversion algorithm
 *  @param pedestal maps, gain maps, dataframes, number of frames, energy map
 */
__global__ void calculate(uint16_t* pede, double* gain, uint16_t* data, uint16_t num, float* energy);

/**
 *  Kernel function to update pedestal maps, call for each level individual
 *  @param 1000+ "dark" frames, number of frames, pedestal maps
 * */
__global__ void calibrate(uint16_t* data, uint16_t num, uint16_t* pede);
