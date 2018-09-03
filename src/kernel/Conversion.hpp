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
            if (gainStageMaps)
                gainStageMaps[i].data[id] = gainStage;
            if (energyMaps)
                energyMaps[i].data[id] = 
                    (adc - statMaps[gainStage * MAPSIZE + id].mean) 
                        / gainMaps[gainStage][id];
            // set energy to zero if pixel is marked false by mask
            if (!isValid) {
                energyMaps[i].data[id] = 0;
            }
            // detect photons
            float sum;
            int maxId;
            int currentIdx;
            bool photonCondition;
            // check for single photon hit
            photonCondition = (energyMaps[i].data[id] > 
                        c * statMaps[gainStage * MAPSIZE + id].stddev);
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
                        photonCondition |= sum > n * c * 
                                statMaps[gainStage * MAPSIZE + id].stddev;
                        // check if center is maximum in cluster
                        photonCondition &= id == maxId;
                    }
                    if (photonCondition) {
                        // photon detected in cluster
                        // copy cluster to cluster vector
                        // TODO: atomic increment
                        auto clusterIdx = clusterVector.used++;
                        int l = 0;
                        int currentIdx;
                        for (int y = -n / 2; y < (n + 1) / 2; ++y) {
                            for (int x = -n / 2; x < (n + 1) / 2; ++x) {
                                currentIdx = id + y * DIMX + x;
                                clusterVector.data[clusterIdx].data[l++] = 
                                    energyMaps[i].data[currentIdx];
                            }
                        }
                    }
                }
            }
            else {
                // no clustering used
                if (photonCondition) {
                    //TODO: calculate photon count, write to photonmap
                }
            }
            // no single photon on single pixel or cluster
            // check if dark pixel and update pedestal
            if (!photonCondition) {
                //TODO: check dark pixel condition, update pedestal
            }
        }
    }
};
