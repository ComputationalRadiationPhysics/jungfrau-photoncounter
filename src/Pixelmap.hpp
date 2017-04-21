#pragma once
#include "Config.hpp"
#include <vector>

template <typename T> class Pixelmap {
private:
    T* buffer;
    std::size_t n;
    bool header;

public:
    using contentT = T;
    static const std::size_t elementSize = sizeof(T);
    Pixelmap(std::size_t n, T* buffer, bool header = true)
        : buffer(buffer), n(n), header(header)
    {
    }
    T& operator()(std::size_t x, std::size_t y, std::size_t n)
    {
        std::size_t index = n * DIMX * DIMY + y * DIMX + x;
        if (header)
            index += n * FRAME_HEADER_SIZE;
        return buffer[index];
    }
    std::size_t getSizeBytes() const
    {
        std::size_t size = DIMX * DIMY * n * sizeof(T);
        if (header)
            size += n * FRAME_HEADER_SIZE;
        return size;
    }
    T* data() const { return buffer; }
    std::size_t getN() const { return n; }
};

using Datamap = Pixelmap<DataType>;
using Gainmap = Pixelmap<GainType>;
using Pedestalmap = Pixelmap<PedestalType>;
using Photonmap = Pixelmap<PhotonType>;
