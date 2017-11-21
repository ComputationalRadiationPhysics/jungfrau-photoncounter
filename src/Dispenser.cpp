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

    alpaka::stream::enqueue(stream, calibration);
    alpaka::wait::wait(stream);

    alpaka::stream::enqueue(stream, correction);
    alpaka::wait::wait(stream);

    alpaka::stream::enqueue(stream, summation);
    alpaka::wait::wait(stream);

    save_image<Photon>(
        static_cast<std::string>("Testframe500.bmp"),
        alpaka::mem::view::getPtrNative(photon_c),
        500ul);
