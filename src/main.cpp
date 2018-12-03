#include "Alpakaconfig.hpp"
#include "Config.hpp"
#include "Dispenser.hpp"
#include "Filecache.hpp"

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */
using Accelerator = GpuCudaRt;//CpuSerial;




void save_pedestal_update_count(std::string path, Pedestal* data)
{
#if (SHOW_DEBUG)
    std::ofstream img;
    img.open(path + ".txt");
    for (std::size_t j = 0; j < 512; j++) {
        for (std::size_t k = 0; k < 1024; k++) {
            double h = double(data[(j * 1024) + k].count);
            img << h << " ";
        }
        img << "\n";
    }
    img.close();
#endif
}

void save_pedestal_stddev(std::string path, Pedestal* data)
{
#if (SHOW_DEBUG)
    std::ofstream img;
    img.open(path + ".txt");
    for (std::size_t j = 0; j < 512; j++) {
        for (std::size_t k = 0; k < 1024; k++) {
            double h = double(data[(j * 1024) + k].stddev);
            img << h << " ";
        }
        img << "\n";
    }
    img.close();
#endif
}

void save_pedestal_m2(std::string path, Pedestal* data)
{
#if (SHOW_DEBUG)
    std::ofstream img;
    img.open(path + ".txt");
    for (std::size_t j = 0; j < 512; j++) {
        for (std::size_t k = 0; k < 1024; k++) {
            double h = double(data[(j * 1024) + k].m2);
            img << h << " ";
        }
        img << "\n";
    }
    img.close();
#endif
}

void save_pedestal_update_mean(std::string path, Pedestal* data)
{
#if (SHOW_DEBUG)
    std::ofstream img;
    img.open(path + ".txt");
    for (std::size_t j = 0; j < 512; j++) {
        for (std::size_t k = 0; k < 1024; k++) {
            double h = double(data[(j * 1024) + k].mean);
            img << h << " ";
        }
        img << "\n";
    }
    img.close();
#endif
}


struct Point {
  uint16_t x, y;
};

class PixelTracker {
private:
  std::vector<Point> positions;
  std::vector<std::vector<double>> input, pedestal, m2, stddev;
  
public:
  PixelTracker(int argc, char* argv[]) {
    int numberOfPixels = (argc - 1) / 2;
    DEBUG("adding " << numberOfPixels << " pixels");
    for(int i = 0; i < numberOfPixels; ++i) {
      DEBUG(atoi(argv[2 * i + 1]) << ":" << atoi(argv[2 * i + 2]));
      addPixel(atoi(argv[2 * i + 1]), atoi(argv[2 * i + 2]));
    }
  }
  
  PixelTracker(std::vector<Point> positions = std::vector<Point>()) : positions(positions) {}
  
  void addPixel(Point position) {
    positions.push_back(position);

    input.resize(input.size() + 1);
    pedestal.resize(input.size() + 1);
    m2.resize(input.size() + 1);
    stddev.resize(input.size() + 1);
  }
  
  void addPixel(uint16_t x, uint16_t y) {
    addPixel({x, y});
  }

  void push_back(FramePackage<PedestalMap, Accelerator, Dim, Size> raw_pedestals, FramePackage<DetectorData, Accelerator, Dim, Size> raw_input, size_t offset) {
    for(int i = 0; i < input.size(); ++i) {
      input[i].push_back(alpaka::mem::view::getPtrNative(raw_input.data)[offset].data[positions[i].y * DIMX + positions[i].x]);
      pedestal[i].push_back(alpaka::mem::view::getPtrNative(raw_pedestals.data)[0][positions[i].y * DIMX + positions[i].x].mean);
      stddev[i].push_back(alpaka::mem::view::getPtrNative(raw_pedestals.data)[0][positions[i].y * DIMX + positions[i].x].stddev);
      m2[i].push_back(alpaka::mem::view::getPtrNative(raw_pedestals.data)[0][positions[i].y * DIMX + positions[i].x].m2);



      //DEBUG("dbug: " << input[i][0] << " " << pedestal[i][0] << " " << m2[i][0] << " " << stddev[i][0]);
    }
  }

  void save() {
    DEBUG("saving " << input.size() << " pixels");
    
    for(int i = 0; i < input.size(); ++i) {
      DEBUG("saving pixel " << std::to_string(positions[i].x) + ":" + std::to_string(positions[i].y));
      
      std::ofstream input_file("input_" + std::to_string(positions[i].x) + "_" + std::to_string(positions[i].y) + ".txt");
      std::ofstream pedestal_file("pedestal_" + std::to_string(positions[i].x) + "_" + std::to_string(positions[i].y) + ".txt");
      std::ofstream m2_file("m2_" + std::to_string(positions[i].x) + "_" + std::to_string(positions[i].y) + ".txt");
      std::ofstream stddev_file("stddev_" + std::to_string(positions[i].x) + "_" + std::to_string(positions[i].y) + ".txt");
      
      for(unsigned int j = 0; j < input[i].size(); ++j) {
        input_file << input[i][j] << " ";
        pedestal_file << pedestal[i][j] << " ";
        m2_file << m2[i][j] << " ";
        stddev_file << stddev[i][j] << " ";
      }
      input_file.flush();
      input_file.close();
      pedestal_file.flush();
      pedestal_file.close();
      m2_file.flush();
      m2_file.close();
      stddev_file.flush();
      stddev_file.close();
    }
  }
};



auto main(int argc, char* argv[]) -> int
{
    // t is used in all debug-messages
    t = Clock::now();

    Filecache* fc = new Filecache(1024UL * 1024 * 1024 * 16);
    DEBUG("filecache created");

    // load maps
    FramePackage<DetectorData, Accelerator, Dim, Size> pedestaldata(
        fc->loadMaps<DetectorData, Accelerator, Dim, Size>(
            "../../jungfrau-photoncounter/data_pool/px_101016/"
            "allpede_250us_1243__B_000000.dat",
            true));
    DEBUG(pedestaldata.numFrames << " pedestaldata maps loaded");

    FramePackage<DetectorData, Accelerator, Dim, Size> data(
        fc->loadMaps<DetectorData, Accelerator, Dim, Size>(
            "../../jungfrau-photoncounter/data_pool/px_101016/"
            "Insu_6_tr_1_45d_250us__B_000000.dat",
            true));
    DEBUG(data.numFrames << " data maps loaded");

    FramePackage<GainMap, Accelerator, Dim, Size> gain(fc->loadMaps<GainMap,
                                                                    Accelerator,
                                                                    Dim,
                                                                    Size>(
        "../../jungfrau-photoncounter/data_pool/px_101016/gainMaps_M022.bin"));
    DEBUG(gain.numFrames << " gain maps loaded");

    FramePackage<MaskMap, Accelerator, Dim, Size> mask(SINGLEMAP);
    mask.numFrames = 0;
    /*(fc->loadMaps<MaskMap, Accelerator, Dim, Size>(
            "../data_pool/px_101016/mask.bin"));
            DEBUG(mask.numFrames << " masking maps loaded");*/
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

    boost::optional<alpaka::mem::buf::Buf<typename Accelerator::DevHost, MaskMap, Dim, Size>> maskPtr;
    if(mask.numFrames == SINGLEMAP)
      maskPtr = mask.data;

    //! @todo: throw this new out
    Dispenser<Accelerator, Dim, Size>* dispenser =
      new Dispenser<Accelerator, Dim, Size>(gain, maskPtr);

    // upload and calculate pedestal data
    dispenser->uploadPedestaldata(pedestaldata);

    FramePackage<PhotonMap, Accelerator, Dim, Size> photon(DEV_FRAMES);
    FramePackage<PhotonSumMap, Accelerator, Dim, Size> sum(DEV_FRAMES / SUM_FRAMES);
    ClusterArray<Accelerator, Dim, Size> clusters;
    FramePackage<EnergyValue, Accelerator, Dim, Size> maxValues(DEV_FRAMES);
    std::size_t offset = 0;
    std::size_t downloaded = 0;
    std::size_t currently_downloaded_frames = 0;

    PixelTracker pt(argc, argv);

    // process data maps
    while (downloaded < data.numFrames) {
        offset = dispenser->uploadData(data, offset);
        if (currently_downloaded_frames = dispenser->downloadData(photon, sum, maxValues, clusters)) {
          pt.push_back(dispenser->downloadPedestaldata(), data, offset - 1);

          if(downloaded + currently_downloaded_frames == 9997)
            exit(EXIT_FAILURE);
          
          downloaded += currently_downloaded_frames;
          DEBUG(downloaded << "/" << data.numFrames << " downloaded; " << offset << " uploaded");
        }
    }
    
    //saveClusters("clusters.bin", clusters);
    //
    //GainStageMap* gainStage = dispenser->downloadGainStages();
    //save_image<GainStageMap>("gainstage", gainStage, 0);

    pt.save();
    
    DriftMap* drift = dispenser->downloadDriftMaps();
    save_image<DriftMap>("tokyodriftmap", drift, 0);

    FramePackage<PedestalMap, Accelerator, Dim, Size> pedestals(dispenser->downloadPedestaldata());
    save_pedestal_update_count("pedestal_updates", alpaka::mem::view::getPtrNative(pedestals.data)[0]);

    for(uint32_t i = 0; i < maxValues.numFrames; ++i)
      DEBUG("max value for frame " << i << ": " << alpaka::mem::view::getPtrNative(maxValues.data)[i]);

    
    auto sizes = dispenser->getMemSize();
    auto free_mem = dispenser->getFreeMem();

    for (std::size_t i = 0; i < sizes.size(); ++i)
        DEBUG("Device #" << i << ": "
                         << (float)free_mem[i] / (float)sizes[i] * 100.0f
                         << "% free of " << sizes[i] << " Bytes");

    return 0;
}
