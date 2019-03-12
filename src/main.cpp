#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"
#include "jungfrau-photoncounter/Dispenser.hpp"
#include "jungfrau-photoncounter/Filecache.hpp"

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */
using Accelerator = GpuCudaRt; // CpuSerial;

struct Point {
    uint16_t x, y;
};

class PixelTracker {
private:
    std::vector<Point> positions;
    std::vector<std::vector<double>> input, pedestal[PEDEMAPS],
        stddev[PEDEMAPS];

public:
    PixelTracker(int argc, char* argv[])
    {
        int numberOfPixels = (argc - 1) / 2;
        DEBUG("adding " << numberOfPixels << " pixels");
        for (int i = 0; i < numberOfPixels; ++i) {
            DEBUG(atoi(argv[2 * i + 1]) << ":" << atoi(argv[2 * i + 2]));
            addPixel(atoi(argv[2 * i + 1]), atoi(argv[2 * i + 2]));
        }
    }

    PixelTracker(std::vector<Point> positions = std::vector<Point>())
        : positions(positions)
    {
    }

    void addPixel(Point position)
    {
        positions.push_back(position);

        if(position.x >= DIMX || position.y >= DIMY) {
          std::cerr << "Pixel out of range!" << std::endl;
          abort();
        }

        input.resize(input.size() + 1);
        for (std::size_t p = 0; p < PEDEMAPS; ++p) {
            pedestal[p].resize(input.size() + 1);
            stddev[p].resize(input.size() + 1);
        }
    }

    void addPixel(uint16_t x, uint16_t y) { addPixel({x, y}); }

    void push_back(
        FramePackage<InitPedestalMap, Accelerator, Dim, Size> init_pedestals,
        FramePackage<PedestalMap, Accelerator, Dim, Size> raw_pedestals,
        FramePackage<DetectorData, Accelerator, Dim, Size> raw_input,
        size_t offset)
    {
        for (int i = 0; i < input.size(); ++i) {
            input[i].push_back(
                  alpaka::mem::view::getPtrNative(raw_input.data)[offset]
                  .data[positions[i].y * DIMX + positions[i].x]);
            DEBUG(
                "input " << alpaka::mem::view::getPtrNative(
                                raw_input.data)[offset]
                                .data[positions[i].y * DIMX + positions[i].x]);
            for (std::size_t p = 0; p < PEDEMAPS; ++p) {
                pedestal[p][i].push_back(alpaka::mem::view::getPtrNative(
                    raw_pedestals
                        .data)[p][positions[i].y * DIMX + positions[i].x]);
                DEBUG("pede["
                      << p << "][" << i << "] "
                      << alpaka::mem::view::getPtrNative(
                             raw_pedestals.data)[p][positions[i].y * DIMX +
                                                    positions[i].x]);
                stddev[p][i].push_back(
                    alpaka::mem::view::getPtrNative(
                        init_pedestals
                            .data)[p][positions[i].y * DIMX + positions[i].x]
                        .stddev);
                DEBUG("stddev["
                      << p << "][" << i << "] "
                      << alpaka::mem::view::getPtrNative(
                             init_pedestals.data)[p][positions[i].y * DIMX +
                                                     positions[i].x]
                             .stddev);
            }
        }
    }

    void save()
    {
        DEBUG("saving " << input.size() << " pixels");

        for (int i = 0; i < input.size(); ++i) {
            DEBUG("saving pixel " << std::to_string(positions[i].x) + ":" +
                                         std::to_string(positions[i].y));

            std::ofstream input_file("input_" + std::to_string(positions[i].x) +
                                     "_" + std::to_string(positions[i].y) +
                                     ".txt");

            std::ofstream pedestal_file[PEDEMAPS];
            std::ofstream stddev_file[PEDEMAPS];

            for (std::size_t p = 0; p < PEDEMAPS; ++p) {
                pedestal_file[p].open("pedestal_" + std::to_string(p) + "_" +
                                      std::to_string(positions[i].x) + "_" +
                                      std::to_string(positions[i].y) + ".txt");

                stddev_file[p].open("stddev_" + std::to_string(p) + "_" +
                                          std::to_string(positions[i].x) + "_" +
                                          std::to_string(positions[i].y) +
                                          ".txt");
            }

            for (unsigned int j = 0; j < input[i].size(); ++j) {
                input_file << input[i][j] << " ";
                for (std::size_t p = 0; p < PEDEMAPS; ++p) {
                    pedestal_file[p] << pedestal[p][i][j] << " ";
                    stddev_file[p] << stddev[p][i][j] << " ";
                }
            }

            input_file.flush();
            input_file.close();

            for (std::size_t p = 0; p < PEDEMAPS; ++p) {
                pedestal_file[p].flush();
                pedestal_file[p].close();
                stddev_file[p].flush();
                stddev_file[p].close();
            }
        }
    }
};

auto main(int argc, char* argv[]) -> int
{
    // t is used in all debug-messages
    t = Clock::now();

    // create a file cache for all input files 
    Filecache* fc = new Filecache(1024UL * 1024 * 1024 * 16);
    DEBUG("filecache created");

    // load maps
    FramePackage<DetectorData, Accelerator, Dim, Size> pedestaldata(
        fc->loadMaps<DetectorData, Accelerator, Dim, Size>(
                                                           //"../../data_pool/px_101016/allpede_250us_1243__B_000000.dat",
            "../../moench_data/"
            "1000_frames_pede_e17050_1_00018_00000.dat",
            true));
    DEBUG(pedestaldata.numFrames << " pedestaldata maps loaded");

    FramePackage<DetectorData, Accelerator, Dim, Size> data(
        fc->loadMaps<DetectorData, Accelerator, Dim, Size>(
                                                           //"../../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat",
            "../../moench_data/"
            "e17050_1_00018_00000_image.dat",
            true));
    DEBUG(data.numFrames << " data maps loaded");

    FramePackage<GainMap, Accelerator, Dim, Size> gain(
        fc->loadMaps<GainMap, Accelerator, Dim, Size>(
            "../../moench_data/moench_gain.bin"
            //"../../data_pool/px_101016/gainMaps_M022.bin"
                                                      ));
    DEBUG(gain.numFrames << " gain maps loaded");

    FramePackage<MaskMap, Accelerator, Dim, Size> mask(SINGLEMAP);
    mask.numFrames = 0;
    delete (fc);

    // print info
#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
    DEBUG("gpu count: "
          << (alpaka::pltf::getDevCount<alpaka::pltf::Pltf<alpaka::dev::Dev<
                  alpaka::acc::AccGpuCudaRt<alpaka::dim::DimInt<1u>,
                                            std::size_t>>>>()));
#endif
    DEBUG("cpu count: " << (alpaka::pltf::getDevCount<
                            alpaka::pltf::Pltf<typename Accelerator::Acc>>()));

    // create empty, optional input mask
    boost::optional<alpaka::mem::buf::
                        Buf<typename Accelerator::DevHost, MaskMap, Dim, Size>>
        maskPtr;
    if (mask.numFrames == SINGLEMAP)
        maskPtr = mask.data;

    //! @todo: throw this new out
    Dispenser<Accelerator, Dim, Size>* dispenser =
        new Dispenser<Accelerator, Dim, Size>(gain, maskPtr);

    // upload and calculate pedestal data
    dispenser->uploadPedestaldata(pedestaldata);

    // allocate space for output data
    FramePackage<EnergyMap, Accelerator, Dim, Size> energy_data(DEV_FRAMES);
    FramePackage<PhotonMap, Accelerator, Dim, Size> photon_data(DEV_FRAMES);
    FramePackage<SumMap, Accelerator, Dim, Size> sum_data(DEV_FRAMES /
                                                                SUM_FRAMES);
    ClusterArray<Accelerator, Dim, Size> clusters_data(30000 * 40000 / 50);
    FramePackage<EnergyValue, Accelerator, Dim, Size> maxValues_data(
        DEV_FRAMES);

    boost::optional<FramePackage<EnergyMap, Accelerator, Dim, Size>&> energy =
        energy_data;
    boost::optional<FramePackage<PhotonMap, Accelerator, Dim, Size>&>
        photon; // = photon_data;
    boost::optional<FramePackage<SumMap, Accelerator, Dim, Size>&> sum =
        sum_data;
    boost::optional<ClusterArray<Accelerator, Dim, Size>&> clusters =
        clusters_data;

    boost::optional<FramePackage<EnergyValue, Accelerator, Dim, Size>&>
        maxValues = maxValues_data;

    std::size_t offset = 0;
    std::size_t downloaded = 0;
    std::size_t currently_downloaded_frames = 0;

    PixelTracker pt(argc, argv);

    ExecutionFlags ef;
    ef.mode = 2; // photon and energy values are calculated
    ef.summation = 1;
    ef.masking = 1;
    ef.maxValue = 1;

    // process data maps
    while (downloaded < data.numFrames) {
        offset = dispenser->uploadData(data, offset, ef);
        if (currently_downloaded_frames = dispenser->downloadData(
                energy, photon, sum, maxValues, clusters)) {

            auto ipdata = dispenser->downloadInitialPedestaldata();
            auto pdata = dispenser->downloadPedestaldata();
            pt.push_back(ipdata, pdata, data, offset - 1);

            if (offset == 10000)
                dispenser->flush();

            downloaded += currently_downloaded_frames;
            DEBUG(downloaded << "/" << data.numFrames << " downloaded; "
                             << offset << " uploaded");
        }
    }

    if (clusters) {
        saveClusters("clusters.txt", *clusters);
        saveClustersBin("clusters.bin", *clusters);
    }

    pt.save();

    DriftMap* drift = dispenser->downloadDriftMaps();
    save_image<DriftMap>("tokyodriftmap", drift, 0);

    auto sizes = dispenser->getMemSize();
    auto free_mem = dispenser->getFreeMem();

    for (std::size_t i = 0; i < sizes.size(); ++i)
        DEBUG("Device #" << i << ": "
                         << (float)free_mem[i] / (float)sizes[i] * 100.0f
                         << "% free of " << sizes[i] << " Bytes");

    return 0;
}
