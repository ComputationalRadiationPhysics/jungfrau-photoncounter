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

    save_image<Photon>(
        static_cast<std::string>("Testframe500.bmp"),
        alpaka::mem::view::getPtrNative(photon_c),
        500ul);
