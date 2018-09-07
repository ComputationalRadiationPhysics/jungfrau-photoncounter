#pragma once
#include <alpaka/alpaka.hpp>
#include "../Config.hpp"

template<typename TDataword>
ALPAKA_FN_ACC auto getAdc(TDataword dataword) { return dataword & 0x3fff; }

template<typename TDataword>
ALPAKA_FN_ACC auto getGainStage(TDataword dataword)
{
    auto g = (dataword & 0xc000) >> 14;
    // map gain stages 0, 1, 3 to 0, 1, 2
    if (g == 3)
        g = 2;
    return g;
}

template<typename TInputMap, typename TOutputMap>
ALPAKA_FN_ACC void copyFrameHeader(TInputMap const& src, TOutputMap& dst)
{
    dst.header = src.header;
}

template <typename TAcc, typename TAdcValue, typename TPedestal>
ALPAKA_FN_ACC void updatePedestal(
        const TAcc& acc, 
        TAdcValue const adc, 
        TPedestal& pedestal
        )
{
    // online algorithm for variance by Welford
    auto& count = pedestal.count;
    auto& mean = pedestal.mean;
    auto& m2 = pedestal.m2;
    auto& var = pedestal.variance; // sample variance
    auto& stddev = pedestal.stddev;

    ++count;
    float delta = adc - mean;
    mean += delta / count;
    float delta2 = adc - mean;
    m2 += delta * delta2;
    var = m2 / (count - 1);
    stddev = alpaka::math::sqrt(acc, var);
}

template <typename TThreadIndex>
ALPAKA_FN_ACC bool indexQualifiesAsClusterCenter(TThreadIndex id)
{
    constexpr auto n = CLUSTER_SIZE;
    return (
        id % DIMX >= n / 2 &&
        id % DIMX <= DIMX - (n + 1) / 2 &&
        id / DIMX >= n / 2 &&
        id / DIMX <= DIMY - (n + 1) / 2);
}

template <
    typename TMap, 
    typename TThreadIndex, 
    typename TSum
    >
ALPAKA_FN_ACC void findClusterSumAndMax(
        TMap const& map, 
        TThreadIndex const id, 
        TSum& sum, 
        TThreadIndex& max
        )
{
    TThreadIndex it = 0;
    max = 0;
    constexpr auto n = CLUSTER_SIZE;
    for (int y = -n / 2; y < (n + 1) / 2; ++y) {
        for (int x = -n / 2; x < (n + 1) / 2; ++x) {
            it = id + y * DIMX + x;
            if (map[max] < map[it])
                max = it;
            sum += map[it];
        }
    }
}

template<typename TAcc, typename TClusterArray>
ALPAKA_FN_ACC
auto& getClusterBuffer(TAcc const& acc, TClusterArray* const clusterArray)
{
    // get next free index of buffer atomically
    auto i = alpaka::atomic::atomicOp<alpaka::atomic::op::Add>(
            acc,
            &clusterArray->size,
            1);
    return clusterArray->clusters[i];
}

template <
    typename TMap, 
    typename TThreadIndex, 
    typename TCluster
    >
ALPAKA_FN_ACC void copyCluster(
        TMap const& map, 
        TThreadIndex const id, 
        TCluster& cluster
        )
{
    constexpr auto n = CLUSTER_SIZE;
    TThreadIndex it;
    TThreadIndex i = 0;
    cluster.x = id % DIMX;
    cluster.y = id / DIMX;
    cluster.frameNumber = map.header.frameNumber;
    for (int y = -n / 2; y < (n + 1) / 2; ++y) {
        for (int x = -n / 2; x < (n + 1) / 2; ++x) {
            it = id + y * DIMX + x;
            cluster.data[i++] = map[it];
        }
    }
}

