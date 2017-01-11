#pragma once
#include <vector>

template <typename T> class Pixelmap {
private:
    std::vector<T> buffer;
    const std::size_t dimX;
    const std::size_t dimY;

public:
    Pixelmap(std::size_t dimX, std::size_t dimY)
        : buffer(dimX * dimY), dimX(dimX), dimY(dimY)
    {
    }
    T* data() { return buffer.data(); }
    T& operator()(std::size_t x, std::size_t y) { return buffer[y * dimX + x]; }
    std::size_t getSizeBytes() const { return buffer.size() * sizeof(T); }
};
