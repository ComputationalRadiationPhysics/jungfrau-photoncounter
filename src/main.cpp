#include "Config.hpp"
#include "Filecache.hpp"
//////////////////////////////TODO: replace with dispenser//
#include "kernel/Calibration.hpp"
#include "kernel/Correction.hpp"
#include "kernel/Summation.hpp"
////////////////////////////////////////////////////////////

#include <iostream>


auto main() -> int
{
    Filecache fc(1024UL * 1024 * 1024 * 5);
    DEBUG("filecache created");

    Maps<Data> pedestaldata(fc.loadMaps<Data>(
        "../data_pool/px_101016/allpede_250us_1243__B_000000.dat", true));
    DEBUG(pedestaldata.numMaps << " pedestaldata maps loaded");

    /*Maps<Data> data(fc.loadMaps<Data>(
        "../data_pool/px_101016/Insu_6_tr_1_45d_250us__B_000000.dat", true));
    DEBUG(data.numMaps << " data maps loaded");*/

    Maps<Data> data(
        fc.loadMaps<Data>("../data_pool/px_101016/1kFrames.dat", true));
    DEBUG(data.numMaps << " data maps loaded");

    Maps<Gain> gain(
        fc.loadMaps<Gain>("../data_pool/px_101016/gainMaps_M022.bin"));
    DEBUG(gain.numMaps << " gain maps loaded");

    // TODO: this will be redone for multiple devices after the kernels work
    // alpaka specific types, CPU serial, make struct out of all of them, vec
    save_image<Data>(
        static_cast<std::string>("TestframeInput.bmp"),
        data.dataPointer,
        1ul);


    using Dim = alpaka::dim::DimInt<1u>;
    using Size = std::size_t;
    using Host = alpaka::acc::AccCpuSerial<Dim, Size>;
    using Acc = alpaka::acc::AccCpuSerial<Dim, Size>;
    using DevHost = alpaka::dev::Dev<Host>;
    using DevAcc = alpaka::dev::Dev<Acc>;
    using PltfHost = alpaka::pltf::Pltf<DevHost>;
    using PltfAcc = alpaka::pltf::Pltf<DevAcc>;
    using WorkDiv = alpaka::workdiv::WorkDivMembers<Dim, Size>;
    using DevStream = alpaka::stream::StreamCpuSync;

    // later use vector -> ringbuffer
    DevAcc const devAcc(alpaka::pltf::getDevByIdx<PltfAcc>(0u));
    DevHost const devHost(alpaka::pltf::getDevByIdx<PltfHost>(0u));

    alpaka::vec::Vec<Dim, Size> const elementsPerThread(static_cast<Size>(1u));

    alpaka::vec::Vec<Dim, Size> const threadsPerBlock(static_cast<Size>(1u));

    alpaka::vec::Vec<Dim, Size> const blocksPerGrid(static_cast<Size>(MAPSIZE));

    WorkDiv const workdiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

    DevStream stream(devAcc);

    // hostbuffer
    alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size> pedestal_d(
        pedestaldata.dataPointer, devHost, pedestaldata.numMaps * MAPSIZE);
    alpaka::mem::view::ViewPlainPtr<DevHost, Data, Dim, Size> data_d(
        data.dataPointer, devHost, data.numMaps * MAPSIZE);
    alpaka::mem::view::ViewPlainPtr<DevHost, Gain, Dim, Size> gain_d(
        gain.dataPointer, devHost, gain.numMaps * MAPSIZE);

    // devicebuffer
    /*
    // Allocate the buffers on the accelerator.
    auto memBufAccA(alpaka::mem::buf::alloc<Val, Size>(devAcc, extent));
    auto memBufAccB(alpaka::mem::buf::alloc<Val, Size>(devAcc, extent));
    auto memBufAccC(alpaka::mem::buf::alloc<Val, Size>(devAcc, extent));

    // Copy Host -> Acc.
    alpaka::mem::view::copy(stream, memBufAccA, memBufHostA, extent);
    alpaka::mem::view::copy(stream, memBufAccB, memBufHostB, extent);
    */

    auto pedestal_c(
        alpaka::mem::buf::alloc<Pedestal, Size>(devAcc, 3 * MAPSIZE));
    auto photon_c(
        alpaka::mem::buf::alloc<Photon, Size>(devAcc, data.numMaps * MAPSIZE));
    auto sum_c(alpaka::mem::buf::alloc<PhotonSum, Size>(
        devAcc, (data.numMaps / SUM_FRAMES) * MAPSIZE));

    CalibrationKernel calibrationKernel;
    CorrectionKernel correctionKernel;
    SummationKernel summationKernel;

    auto const calibration(
        alpaka::exec::create<Acc>(workdiv,
                                  calibrationKernel,
                                  alpaka::mem::view::getPtrNative(pedestal_d),
                                  alpaka::mem::view::getPtrNative(pedestal_c),
                                  3000u));
    auto const correction(
        alpaka::exec::create<Acc>(workdiv,
                                  correctionKernel,
                                  alpaka::mem::view::getPtrNative(data_d),
                                  alpaka::mem::view::getPtrNative(pedestal_c),
                                  alpaka::mem::view::getPtrNative(gain_d),
                                  1000u,
                                  alpaka::mem::view::getPtrNative(photon_c)));
    auto const summation(
        alpaka::exec::create<Acc>(workdiv,
                                  summationKernel,
                                  alpaka::mem::view::getPtrNative(photon_c),
                                  SUM_FRAMES,
                                  1000u,
                                  alpaka::mem::view::getPtrNative(sum_c)));

    DEBUG("jetzt kernelts");
    alpaka::stream::enqueue(stream, calibration);
    alpaka::wait::wait(stream);
    DEBUG(alpaka::mem::view::getPtrNative(pedestal_c)[18].value);
    DEBUG("jetzt kernelts");
    alpaka::stream::enqueue(stream, correction);
    alpaka::wait::wait(stream);
    DEBUG(alpaka::mem::view::getPtrNative(photon_c)[18]);
    DEBUG("jetzt kernelts");
    alpaka::stream::enqueue(stream, summation);
    alpaka::wait::wait(stream);
    DEBUG(alpaka::mem::view::getPtrNative(sum_c)[18]);

    save_image<Photon>(
        static_cast<std::string>("Testframe.bmp"),
        alpaka::mem::view::getPtrNative(photon_c),
        1ul);
    save_image<Photon>(
        static_cast<std::string>("Testframe500.bmp"),
        alpaka::mem::view::getPtrNative(photon_c),
        500ul);
    ///////////////////////////////////////////////////////////////////////

    return 0;
}
