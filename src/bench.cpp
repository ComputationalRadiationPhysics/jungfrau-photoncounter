#include "bench.hpp"
#include "confgen.hpp"

#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"
#include "jungfrau-photoncounter/Dispenser.hpp"

#include "jungfrau-photoncounter/Debug.hpp"
#include <chrono>
#include <unordered_map>
#include <vector>

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */

template <std::size_t MAPSIZE>
using Accelerator = CpuSerial<MAPSIZE>; // CpuOmp2Blocks< // GpuCudaRt<
// MAPSIZE>; // CpuSerial<MAPSIZE>;//GpuCudaRt<MAPSIZE>;//GpuCudaRt<MAPSIZE>;
// // CpuSerial;

//#define MOENCH

#ifdef MOENCH
// using Config = MoenchConfig;
std::string pedestalPath = "../../../../moench_data/1000_frames_pede_e17050_1_00018_00000.dat";
std::string gainPath = "../../../../moench_data/moench_gain.bin";
std::string dataPath = "../../../../moench_data/e17050_1_00018_00000_image.dat";
#else
// using Config = JungfrauConfig;
std::string pedestalPath = "../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat";
std::string gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin";
std::string dataPath = "../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat";
#endif

constexpr auto framesPerStageG0 = Values<std::size_t, 1000>();
constexpr auto framesPerStageG1 = Values<std::size_t, 1000>();
constexpr auto framesPerStageG2 = Values<std::size_t, 1000>();
constexpr auto dimX = Values<std::size_t, 1024>();
constexpr auto dimY = Values<std::size_t, 512>();
constexpr auto sumFrames = Values<std::size_t, 2, 10, 20, 100>();
constexpr auto devFrames = Values<std::size_t, 10, 100, 1000>();
constexpr auto movingStatWindowSize = Values<std::size_t, 100>();
constexpr auto clusterSize = Values<std::size_t, 2, 3, 7, 11>();
constexpr auto cs = Values<std::size_t, 5>();

constexpr auto parameterSpace = framesPerStageG0 * framesPerStageG1 * framesPerStageG2 * dimX *
                                dimY * sumFrames * devFrames * movingStatWindowSize * clusterSize *
                                cs;

template <class Tuple>
struct ConfigFrom {
    using G0 = Get_t<0, Tuple>;
    using G1 = Get_t<1, Tuple>;
    using G2 = Get_t<2, Tuple>;
    using DimX = Get_t<3, Tuple>;
    using DimY = Get_t<4, Tuple>;
    using SumFrames = Get_t<5, Tuple>;
    using DevFrames = Get_t<6, Tuple>;
    using MovingStatWinSize = Get_t<7, Tuple>;
    using ClusterSize = Get_t<8, Tuple>;
    using C = Get_t<9, Tuple>;
    using Result =
        DetectorConfig<G0::value, G1::value, G2::value, DimX::value, DimY::value, SumFrames::value,
                       DevFrames::value, MovingStatWinSize::value, ClusterSize::value, C::value>;
};

using Duration = std::chrono::milliseconds;
using Timer = std::chrono::high_resolution_clock;

template <class Tuple>
std::vector<Duration> benchmark(int iterations, ExecutionFlags flags,
                                const std::string& pedestalPath, const std::string& gainPath,
                                const std::string& dataPath) {
    using Config = typename ConfigFrom<Tuple>::Result;
    using ConcreteAcc = Accelerator<Config::MAPSIZE>;
    auto benchmarkingInput = setUp<Config, ConcreteAcc>(flags, pedestalPath, gainPath, dataPath);
    auto dispenser = calibrate(benchmarkingInput);
    std::vector<Duration> results;
    results.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
        auto t0 = Timer::now();
        bench(dispenser, benchmarkingInput);
        auto t1 = Timer::now();
        results.push_back(std::chrono::duration_cast<Duration>(t1 - t0));
    }
    return results;
}

using BenchmarkFunction = std::vector<Duration> (*)(int, ExecutionFlags, const std::string&,
                                                    const std::string&, const std::string&);

static std::unordered_map<int, BenchmarkFunction> benchmarks;

void registerBenchmarks(int, Empty) {}

template <class List>
void registerBenchmarks(int x, const List&) {
    using H = typename List::Head;
    using T = typename List::Tail;
    using F = typename Flatten<Tuple<>, H>::Result;
    benchmarks[x] = benchmark<F>;
    registerBenchmarks(x + 1, T{});
}

int main() {
    registerBenchmarks(0, parameterSpace);
    std::cout << benchmarks.size() << "\n";
    return 0;
}
