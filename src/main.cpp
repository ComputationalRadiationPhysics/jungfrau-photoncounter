#include "jungfrau-photoncounter/Alpakaconfig.hpp"
#include "jungfrau-photoncounter/Config.hpp"
#include "jungfrau-photoncounter/Dispenser.hpp"
#include "jungfrau-photoncounter/Filecache.hpp"

/**
 * only change this line to change the backend
 * see Alpakaconfig.hpp for all available
 */
using Accelerator = GpuCudaRt;

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
            "../../data_pool/px_101016/"
            "allpede_250us_1243__B_000000.dat",
            true));
    DEBUG(pedestaldata.numFrames << " pedestaldata maps loaded");

    FramePackage<DetectorData, Accelerator, Dim, Size> data(
        fc->loadMaps<DetectorData, Accelerator, Dim, Size>(
            "../../data_pool/px_101016/"
            "Insu_6_tr_1_45d_250us__B_000000.dat",
            true));
    DEBUG(data.numFrames << " data maps loaded");

    FramePackage<GainMap, Accelerator, Dim, Size> gain(
        fc->loadMaps<GainMap, Accelerator, Dim, Size>(
            "../../data_pool/px_101016/gainMaps_M022.bin"));
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
    // ClusterArray<Accelerator, Dim, Size> clusters_data(
    //    150000 * DEV_FRAMES); // MAX_CLUSTER_NUM_USER * DEV_FRAMES);
    FramePackage<EnergyValue, Accelerator, Dim, Size> maxValues_data(
        DEV_FRAMES);

    // create placeholder for output data
    boost::optional<FramePackage<EnergyMap, Accelerator, Dim, Size>&> energy =
        energy_data;
    boost::optional<FramePackage<PhotonMap, Accelerator, Dim, Size>&> photon =
        photon_data;
    boost::optional<FramePackage<SumMap, Accelerator, Dim, Size>&> sum =
        sum_data;
    boost::optional<ClusterArray<Accelerator, Dim, Size>&>
        clusters; // = clusters_data;
    boost::optional<FramePackage<EnergyValue, Accelerator, Dim, Size>&>
        maxValues = maxValues_data;

    std::size_t offset = 0;
    std::size_t downloaded = 0;
    std::size_t currently_downloaded_frames = 0;

    ExecutionFlags ef;
    ef.mode = 1; // photon and energy values are calculated
    ef.summation = 1;
    ef.masking = 1;
    ef.maxValue = 1;

    // process data maps
    while (downloaded < data.numFrames) {
        offset = dispenser->uploadData(data, offset, ef);
        if (currently_downloaded_frames = dispenser->downloadData(
                energy, photon, sum, maxValues, clusters)) {
            downloaded += currently_downloaded_frames;
            
            // Here would be a good place to output or process the data.
            // An example on how to access the data is shown below:
            /*
            // process sums
            SumMap* sum_output =
                alpaka::mem::view::getPtrNative(sum->data);
            for (unsigned int frame = 0; frame < sum->numFrames;
                 ++frame) {
                for (unsigned int y = 0; y < DIMY; ++y) {
                    for (unsigned int x = 0; x < DIMX; ++x) {
                        unsigned int index = DIMX * y + x;
                        printf("frame %d: sum[%d][%d] = %d\n",
                               frame,
                               y,
                               x,
                               sum_output[frame].data[index]);
                    }
                }
            }
           
            // process energy data
            EnergyMap* energy_output =
                alpaka::mem::view::getPtrNative(energy->data);
            for(unsigned int frame = 0; frame < energy->numFrames; ++frame) {
              for(unsigned int y = 0; y < DIMY; ++y) {
                for(unsigned int x = 0; x < DIMX; ++x) {
                  unsigned int index = DIMX * y + x;
                  printf("frame %d: energy[%d][%d] = %f\n", frame, y, x,
energy_output[frame].data[index]);
                }
              }
            }

            // process photon data
            PhotonMap* photon_output =
                alpaka::mem::view::getPtrNative(photon->data);
            for(unsigned int frame = 0; frame < photon->numFrames; ++frame) {
              for(unsigned int y = 0; y < DIMY; ++y) {
                for(unsigned int x = 0; x < DIMX; ++x) {
                  unsigned int index = DIMX * y + x;
                  printf("frame %d: photon[%d][%d] = %d\n", frame, y, x,
photon_output[frame].data[index]);
                }
              }
            }
            */

            DEBUG(downloaded << "/" << data.numFrames << " downloaded; "
                             << offset << " uploaded");
        }
    }

    // save clusters if they exist
    if (clusters)
        saveClusters("clusters.bin", *clusters);

    // print out some info anout the GPU memory
    auto sizes = dispenser->getMemSize();
    auto free_mem = dispenser->getFreeMem();

    for (std::size_t i = 0; i < sizes.size(); ++i)
        DEBUG("Device #" << i << ": "
                         << (float)free_mem[i] / (float)sizes[i] * 100.0f
                         << "% free of " << sizes[i] << " Bytes");

    return 0;
}
