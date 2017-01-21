#pragma once
#include <cstdint>
#include <vector>

#include <iostream>
#define DEBUG(msg) (std::cout << __FILE__ << "["<< __LINE__ << "]:\t" << msg << std::endl)

template <typename T> class Pixelmap {
private:
    T* buffer;
    std::size_t dimX;
    std::size_t dimY;

public:
    using contentT = T;
    static const std::size_t elementSize = sizeof(T);
    Pixelmap(std::size_t dimX, std::size_t dimY, T* buffer)
        : buffer(buffer), dimX(dimX), dimY(dimY)
    {
    }
    T& operator()(std::size_t x, std::size_t y) { return buffer[y * dimX + x]; }
    std::size_t getSizeBytes() const { return dimX * dimY * sizeof(T); }
    T* data() const { return buffer; }
};

using Datamap = Pixelmap<uint16_t>;
using Gainmap = Pixelmap<double>;
using Pedestalmap = Pixelmap<uint16_t>;
using Photonmap = Pixelmap<uint16_t>;
