#pragma once
#include "Pixelmap.h"
#include <cstdint>

class Pedestalmap : public Pixelmap<uint16_t> {
    using Pixelmap::Pixelmap;
};
