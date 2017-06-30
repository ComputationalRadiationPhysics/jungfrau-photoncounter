#pragma once

static const size_t packagesize = 500;
static const size_t framesPerStage = 1000;
static const size_t frameoffset = 8;

struct pedestal{
    uint32_t counter;
    uint16_t value;
    uint32_t movAvg;

    pedestal() : counter(0), value(0), movAvg(0) {}
};
