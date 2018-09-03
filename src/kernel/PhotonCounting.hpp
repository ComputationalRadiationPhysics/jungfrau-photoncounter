#include "../Config.hpp"

struct PhotonCountingKernel {
    template <typename TAcc,
              typename TEnergyMap,
              typename TGainStageMap,
              typename TStatistics,
              typename TNumFrames,
              typename TPhotonMap,
              typename TClusterVector,
              typename TStandardDeviations,
              typename TClusterSize
              >
    ALPAKA_FN_ACC auto operator()(TAcc const& acc,
                                  TEnergyMap const* const energyMaps,
                                  TGainStageMap const* const gainStageMaps,
                                  TStatistics const* const statMaps,
                                  TNumFrames const numFrames,
                                  TPhotonMap* const photonMaps,
                                  bool useCpf = false,
                                  TClusterVector* const clusterVectors = nullptr,
                                  TStandardDeviations const c = 5.f,
                                  TClusterSize const n = 3
                                  ) const -> void
    {
        auto const globalThreadIdx =
            alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc);
        auto const globalThreadExtent =
            alpaka::workdiv::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc);

        auto const linearizedGlobalThreadIdx =
            alpaka::idx::mapIdx<1u>(globalThreadIdx, globalThreadExtent);

        auto id = linearizedGlobalThreadIdx[0u];

        for (std::size_t i = 0; i < num; ++i) {
            // first thread copies frame header
            if (id == 0) {
                photonMaps[i].header = energyMaps[i].header;
            }
            auto gainStage = gainStageMaps[i].data[id];
            // cluster finding algorithm
            if (useCpf) {
            }
            // normal photon counting
            else {
                if 
            }
        }
    }
};
