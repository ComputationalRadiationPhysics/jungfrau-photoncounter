#pragma once

static const size_t PACKAGESIZE = 500;
static const size_t MAPSZIE = 512 * 1024;
static const size_t FRAMESPERSTAGE = 1000;
static const size_t FRAMEOFFSET = 8;
static const size_t BEAMCONST = 6.2;
static const size_t PHOTONCONST = (1. / 12.4);

struct pedestal{
    uint32_t counter;
    uint16_t value;
    uint32_t movAvg;

    pedestal() : counter(0), value(0), movAvg(0) {}
};
