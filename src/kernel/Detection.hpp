#include "../Config.hpp"

struct DetectionKernel {
    template <typename TAcc,
              typename TEnergy
              typename TPedestal,
              typename TNum,
              typename TClusterSize
              typename TClusterVecCapacity,
              typename TClusterVecSize,
              typename TClusterVecData,
              typename TStandardDeviations
              >
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TEnergy* const energyMaps
                                  TPedestal* const pede,
                                  TNum const num,
                                  TClusterSize const clusterSize,
                                  TClusterVecCapacity const maxClusters,
                                  TClusterVecSize* const numClusters,
                                  TClusterVecData* const clusters,
                                  TStandardDeviations const stddevs
                                  ) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        if (id % DIMX >= clusterSize / 2 &&
            id % DIMX <= (clusterSize + 1) / 2 &&
            id / DIMX >= clusterSize / 2 &&
            id / DIMX <= DIMY - (clusterSize + 1) / 2) {
                float sum;
                float max;
                auto maxId = id;
                for (std::size_t frame = 0; frame < num; ++frame) {
                    sum = 0;
                    max = -1.0f/0.0f;
                    maxId = -1;
                    for (char i = -clusterSize; i <= clusterSize; ++i) {
                        for (char j = -clusterSize; j <= clusterSize; ++j) {
                            auto currentIdx = id + i * DIMX + j;
                            float energy = energyMaps[frame].data[currentIdx];
                            sum += energy;
                            if (max < energy) {
                                max = energy;
                                maxId = currentIdx;
                            }
                        }
                    }
                    if (sum > clusterSize * stddevs * pede[id].stddev) {
                        if (maxId == id) {
                            // current pixel is cluster center
                        }
                        else {
                            // current pixel is part of cluster, but not center
                        }
                    } 
                    else {
                        // current pixel might be dark pixel
                    }
                    
                }
        }
    }
};
