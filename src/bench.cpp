#include <benchmark/benchmark.h>

#include "bench.hpp"

#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"
#include "jungfrau-photoncounter/Dispenser.hpp"

#include "jungfrau-photoncounter/Debug.hpp"

template <typename Config, template <std::size_t> class Accelerator>
void BM_Dispenser_UploadData(
    benchmark::State& state,
    Dispenser<Config, Accelerator>* dispenser,
    BenchmarkingInput<Config, Accelerator<Config::MAPSIZE>>* benchmarkingInput)
{
    for (auto _ : state)
        bench<Config, Accelerator>(*dispenser, *benchmarkingInput);
}

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */

template <std::size_t MAPSIZE>
using Accelerator =
    GpuCudaRt<MAPSIZE>; // CpuOmp2Blocks< // GpuCudaRt<
// MAPSIZE>; // CpuSerial<MAPSIZE>;//GpuCudaRt<MAPSIZE>;//GpuCudaRt<MAPSIZE>;
// // CpuSerial;

//#define MOENCH

#ifdef MOENCH
using Config = MoenchConfig;
std::string pedestalPath =
    "../../../../moench_data/1000_frames_pede_e17050_1_00018_00000.dat";
std::string gainPath = "../../../../moench_data/moench_gain.bin";
std::string dataPath = "../../../../moench_data/e17050_1_00018_00000_image.dat";
#else
using Config = JungfrauConfig;
std::string pedestalPath =
    "../../../data_pool/px_101016/allpede_250us_1243__B_000000.dat";
std::string gainPath = "../../../data_pool/px_101016/gainMaps_M022.bin";
std::string dataPath =
    "../../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat";
#endif

using ConcreteAcc = Accelerator<Config::MAPSIZE>;

auto main(int argc, char* argv[]) -> int
{
    typename Config::ExecutionFlags flags = {1, 1, 1, 0};

    BenchmarkingInput<Config, Accelerator<Config::MAPSIZE>> benchmarkingInput =
        SetUp<Config, Accelerator<Config::MAPSIZE>>(
            flags, pedestalPath, gainPath, dataPath);
    Dispenser<Config, Accelerator> dispenser =
        calibrate<Config, Accelerator>(benchmarkingInput);

    

    benchmark::RegisterBenchmark("Benchmark",
                                 &BM_Dispenser_UploadData<Config, Accelerator>,
                                 &dispenser,
                                 &benchmarkingInput)->UseRealTime();

    

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
}
