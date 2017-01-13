#pragma once
#include <cstdint>

/**
 *  Kernel function with conversion algorithm
 *  @param pedestal maps, gain maps, dataframes, number of frames, energy map
 */
void calculate(uint16_t* pede, double* gain, uint16_t* data, uint16_t num, uint16_t energy);

/**
 *  Kernel function to update pedestal maps
 *  @param 1000+ "dark" frames, number of frames, pedestal maps
 * */
void calibrate(uint16_t* data, uint16_t num, uint16_t* pede);
