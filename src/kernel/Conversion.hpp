#include "../Config.hpp"

struct ConversionKernel {
    template <typename TAcc,
              typename TDetectorData,
              typename TGainMap,
              typename TStatistics,
              typename TNumFrames,
              typename TGainStageMap,
              typename TEnergyMap,
              typename TPhotonMap,
              typename TMask,
              typename TNumStdDevs,
              typename TClusterSize,
              typename TClusterVector
              >
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TDetectorData const* const detectorData,
                                  TGainMap const* const gainMaps,
                                  TStatistics const* const statMaps,
                                  TNumFrames const numFrames,
                                  TGainStageMap* const gainStageMaps,
                                  TEnergyMap* const energyMaps,
                                  TPhotonMap* const photonMaps,
                                  TMask* const manualMask,
                                  TMask* const mask,
                                  bool useCpf = false,
                                  TNumStdDevs c = 5,
                                  TClusterSize n = 3,
                                  TClusterVector* const clusterVector
                                  ) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];
        bool isValid = manualMask[id] && mask[id];
        // for each frame to be processed
        for (std::size_t i = 0; i < num; ++i) {
            auto dataword = detectorData[i].data[id];
            // first thread copies frame header to optional output
            if (id == 0) {
                auto header = detectorData[i].header;
                if (energyMaps)
                    energyMaps[i].header = header;
                if (photonMaps)
                    photonMaps[i].header = header;
                if (gainStageMaps)
                    gainStageMaps[i].header = header;
            }
            std::uint16_t adc = dataword & 0x3fff;
            char gainStage = (dataword & 0xc000) >> 14;
            // map gain stages from 0, 1, 3 to 0, 1, 2
            if (gainStage == 3) {
                gainStage = 2;
            }
            // calculate energy
            auto energy = (adc - statMaps[gainStage][id].mean) 
                        / gainMaps[gainStage][id];
            // set energy to zero if marked invalid / false by mask
            if (!isValid)
                energy = 0;
            // set values in optional maps
            if (gainStageMaps)
                gainStageMaps[i].data[id] = gainStage;
            if (energyMaps)
                energyMaps[i].data[id] = energy;
            // detect photons
            float sum;
            int maxId;
            int currentIdx;
            bool photonCondition;
            auto& mean = statMaps[gainStage][id].mean;
            auto& stddev = statMaps[gainStage][id].stddev;
            // check for single photon hit
            photonCondition = (energy > c * stddev);
            if (useCpf) {
                // check if pixel is valid cluster center
                if (id % DIMX >= n / 2 &&
                    id % DIMX <= w - (n + 1) / 2 &&
                    id / DIMX >= n / 2 &&
                    id / DIMX <= DIMY - (n + 1) / 2) {
                        sum = 0;
                        maxId = 0;
                        for (int y = -n / 2; y < (n + 1) / 2; ++y) {
                            for (int x = -n / 2; x < (n + 1) / 2; ++x) {
                                currentIdx = id + y * DIMX + x;
                                if (energyMaps[i].data[maxId] <= 
                                        energyMaps[i].data[currentIdx])
                                    maxId = currentIdx;
                                sum += energyMaps[i].data[currentIdx];
                            }
                        }
                        // check if sum of charges is high enough
                        photonCondition |= sum > n * c * stddev;
                        // check if center is maximum in cluster
                        photonCondition &= id == maxId;
                    }
                    if (photonCondition) {
                        // photon detected in cluster
                        // obtain free buffer for cluster (atomically)
                        auto clusterIdx =
                        alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(
                                acc,
                                &clusterVectors[i].used,
                                1);
                        // copy cluster information
                        clusterVectors[i].frameNumber = header.frameNumber;
                        clusterVectors[i].x = id % DIMX;
                        clusterVectors[i].y = id / DIMX;
                        // fill cluster buffer with energy data
                        int l = 0;
                        int currentIdx;
                        for (int y = -n / 2; y < (n + 1) / 2; ++y) {
                            for (int x = -n / 2; x < (n + 1) / 2; ++x) {
                                currentIdx = id + y * DIMX + x;
                                clusterVectors[i].data[clusterIdx].data[l++] = 
                                    energyMaps[i].data[currentIdx];
                            }
                        }
                    }
                }
            }
            else {
                // no clustering used, optionally count photons
                if (photonMaps && photonCondition) {
                        photonsMaps[i].data[id] = energy / BEAMCONST;
                }
            }
            if (!photonCondition) {
                // no photons detected
                // check if dark pixel
                if (pedestal - c * stddev <= adc &&
                        adc <= pedestal + c * stddev) {
                    // dark pixel found, update statistics for pixel
                    // algorithm by Welford
                    float delta, delta2;
                    auto& count = statMaps[gainStage][id].count;
                    auto& variance = statMaps[gainStage][id].variance;
                    auto& m2 = statMaps[gainStage][id].m2;
                    ++count;
                    delta = adc - mean;
                    mean += delta / count;
                    delta2 = adc - mean;
                    m2 += delta * delta2;
                    variance = m2 / count;
                    stddev = sqrt(variance);
                }
            }
        }
    }
};
