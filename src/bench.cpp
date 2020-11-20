#include "bench.hpp"
#include "check.hpp"
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

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template <std::size_t TMapSize> using Accelerator = GpuCudaRt<TMapSize>;
#else
#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
template <std::size_t TMapSize> using Accelerator = CpuOmp2Blocks<TMapSize>;
#else
template <std::size_t TMapSize> using Accelerator = CpuSerial<TMapSize>;
#endif
#endif

/*
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
template <std::size_t TMapSize> using Accelerator = GpuCudaRtSync<TMapSize>;
#else
template <std::size_t TMapSize> using Accelerator = CpuSerial<TMapSize>;
#endif*/

// CpuOmp2Blocks<MAPSIZE>;
// CpuTbbRt<MAPSIZE>;
// CpuSerial<MAPSIZE>;
// GpuCudaRt<MAPSIZE>;
// GpuHipRt<MAPSIZE>;

constexpr auto framesPerStageG0 = Values<std::size_t, 1000>();
constexpr auto framesPerStageG1 = Values<std::size_t, 1000>();
constexpr auto framesPerStageG2 = Values<std::size_t, 999>();
constexpr auto dimX = Values<std::size_t, 1024>();
constexpr auto dimY = Values<std::size_t, 512>();
constexpr auto sumFrames = Values<std::size_t, 10>(); // 2, 10, 20, 100>();
constexpr auto devFrames =
    Values<std::size_t, 100>(); // 10, 100>(); //, 1000>();
constexpr auto movingStatWindowSize = Values<std::size_t, 100>();
constexpr auto clusterSize = Values<std::size_t, 3>(); // 2, 3, 7, 11>();
constexpr auto cs = Values<std::size_t, 5>();

constexpr auto parameterSpace =
    framesPerStageG0 * framesPerStageG1 * framesPerStageG2 * dimX * dimY *
    sumFrames * devFrames * movingStatWindowSize * clusterSize * cs;

template <class Tuple> struct ConfigFrom {
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
      DetectorConfig<G0::value, G1::value, G2::value, DimX::value, DimY::value,
                     SumFrames::value, DevFrames::value,
                     MovingStatWinSize::value, ClusterSize::value, C::value>;
};
using Duration = std::chrono::nanoseconds;
using Timer = std::chrono::high_resolution_clock;

template <class Tuple>
std::vector<Duration> benchmark(unsigned int iterations, ExecutionFlags flags,
                                const std::string &pedestalPath,
                                const std::string &gainPath,
                                const std::string &dataPath, double beamConst,
                                ResultCheck resultCheck) {
  using Config = typename ConfigFrom<Tuple>::Result;
  using ConcreteAcc = Accelerator<Config::MAPSIZE>;

  std::cout << "Parameters: sumFrames=" << Config::SUM_FRAMES
            << "; devFrames=" << Config::DEV_FRAMES
            << "; clusterSize=" << Config::CLUSTER_SIZE << "\n";
  std::cout << "Flags: mode=" << static_cast<int>(flags.mode)
            << "; summation=" << static_cast<int>(flags.summation)
            << "; masking=" << static_cast<int>(flags.masking)
            << "; maxValue=" << static_cast<int>(flags.maxValue) << "\n";

  auto benchmarkingInput = setUp<Config, ConcreteAcc>(
      flags, pedestalPath, gainPath, dataPath, beamConst);
  std::vector<Duration> results;
  results.reserve(iterations);
  for (unsigned int i = 0; i < iterations; ++i) {
    if (benchmarkingInput.clusters) {
      benchmarkingInput.clusters->used = 0;
      alpakaNativePtr(benchmarkingInput.clusters->usedPinned)[0] = 0;
    }

    DEBUG("Start calibration");

    auto dispenser = calibrate(benchmarkingInput);

    DEBUG("Start run", i);

    auto t0 = Timer::now();
    bench(dispenser, benchmarkingInput);
    auto t1 = Timer::now();

    DEBUG("Finishing run", i);

    results.push_back(std::chrono::duration_cast<Duration>(t1 - t0));
  }

  // check result if requested
  std::cout << "Checking energy if needed ..." << std::endl;
  if (!checkResult(benchmarkingInput.energy, resultCheck.energyPath))
    std::cerr << "Energy result mismatch!\n";

  std::cout << "Checking photons if needed ..." << std::endl;
  if (!checkResult(benchmarkingInput.photons, resultCheck.photonPath))
    std::cerr << "Photon result mismatch!\n";

  std::cout << "Checking sums if needed ..." << std::endl;
  if (!checkResult(benchmarkingInput.sum, resultCheck.sumPath))
    std::cerr << "Sum result mismatch!\n";

  std::cout << "Checking max values if needed ..." << std::endl;
  if (!checkResultRaw(benchmarkingInput.maxValues, resultCheck.maxValuesPath))
    std::cerr << "Maximum value result mismatch!\n";

  std::cout << "Checking clusters if needed ..." << std::endl;
  if (!checkClusters<Config, ConcreteAcc>(benchmarkingInput.clusters,
                                          resultCheck.clusterPath))
    std::cerr << "Cluster result mismatch!\n";

  return results;
}

using BenchmarkFunction = std::vector<Duration> (*)(
    unsigned int, ExecutionFlags, const std::string &, const std::string &,
    const std::string &, double, ResultCheck resultCheck);

static std::unordered_map<int, BenchmarkFunction> benchmarks;

void registerBenchmarks(int, Empty) {}

template <class List> void registerBenchmarks(int x, const List &) {
  using H = typename List::Head;
  using T = typename List::Tail;
  using F = typename Flatten<Tuple<>, H>::Result;
  benchmarks[x] = benchmark<F>;
  registerBenchmarks(x + 1, T{});
}

int main(int argc, char *argv[]) {
  // check command line parameters
  if (argc < 11 || argc > 17) {
    std::cerr << "Usage: bench <benchmark id> <iteration count> "
                 "<beamConst> <mode> <masking> <max values> <summation> "
                 "<pedestal path> <gain path> <data path> [output prefix] "
                 "[energy reference result path] [photon reference result "
                 "path] [max value reference result path] [sum reference "
                 "result path] [cluster reference result path]\n";
    abort();
  }

  // initialize parameters
  int benchmarkID = std::atoi(argv[1]);
  unsigned int iterationCount = static_cast<unsigned int>(std::atoi(argv[2]));
  double beamConst = std::atof(argv[3]);
  ExecutionFlags ef;
  ef.mode = static_cast<std::uint8_t>(std::atoi(argv[4]));
  ef.masking = static_cast<std::uint8_t>(std::atoi(argv[5]));
  ef.maxValue = static_cast<std::uint8_t>(std::atoi(argv[6]));
  ef.summation = static_cast<std::uint8_t>(std::atoi(argv[7]));

  std::string pedestalPath(argv[8]);
  std::string gainPath(argv[9]);
  std::string dataPath(argv[10]);
  std::string outputPath((argc >= 12) ? std::string(argv[11]) + "_" : "");

  // create output path suffix
  outputPath +=
      std::to_string(benchmarkID) + "_" + std::to_string(iterationCount) + "_" +
      std::to_string(ef.mode) + "_" + std::to_string(ef.masking) + "_" +
      std::to_string(ef.maxValue) + "_" + std::to_string(ef.summation) + "_" +
      pedestalPath + "_" + gainPath + "_" + dataPath + ".txt";

  // escape suffix
  std::transform(outputPath.begin(), outputPath.end(), outputPath.begin(),
                 [](char c) -> char { return (c == '/') ? ' ' : c; });

  // store reference result pathes (enter "_" for none)
  ResultCheck resultCheck;
  if (argc > 12)
    resultCheck.energyPath = argv[12];
  if (argc > 13)
    resultCheck.photonPath = argv[13];
  if (argc > 14)
    resultCheck.maxValuesPath = argv[14];
  if (argc > 15)
    resultCheck.sumPath = argv[15];
  if (argc > 16)
    resultCheck.clusterPath = argv[16];

  // register benchmarks
  registerBenchmarks(0, parameterSpace);
  std::cout << "Registered " << benchmarks.size() << " benchmarks. \n";

  // check benchmark ID
  if (benchmarkID < 0 ||
      static_cast<unsigned int>(benchmarkID) >= benchmarks.size()) {
    std::cerr << "Benchmark ID out of range. \n";
    abort();
  }

  // open output file
  std::ofstream outputFile(outputPath);
  if (!outputFile.is_open()) {
    std::cerr << "Couldn't open output file " << outputPath << "\n";
    abort();
  }

  // run benchmark
  auto results =
      benchmarks[benchmarkID](iterationCount, ef, pedestalPath, gainPath,
                              dataPath, beamConst, resultCheck);

  // store results
  for (const auto &r : results)
    outputFile << r.count() << " ";

  outputFile.flush();
  outputFile.close();

  return 0;
}
