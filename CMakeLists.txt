cmake_minimum_required(VERSION 3.7)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(_TARGET_NAME "jungfrau-photoncounter")
project(${_TARGET_NAME})

SET(alpaka_DIR "~/Files/Libs/alpaka") 

find_package(Boost QUIET COMPONENTS unit_test_framework)
if(NOT Boost_UNIT_TEST_FRAMEWORK_FOUND)
    message(FATAL_ERROR i
        "Required alpaka dependency Boost.Test could not be found!")
else()
    list(APPEND _INCLUDE_DIRECTORIES_PRIVATE ${Boost_INCLUDE_DIRS})
    list(APPEND _LINK_LIBRARIES_PRIVATE ${Boost_LIBRARIES})

    if(NOT Boost_USE_STATIC_LIBS)
        list(APPEND _DEFINITIONS_PRIVATE "-DBOOST_TEST_DYN_LINK")
    endif()
endif()


find_package(alpaka REQUIRED)
if(NOT alpaka_FOUND)
    message(WARNING 
        "Required alpaka test common dependency alpaka could not be found!")
else()
    include("${_ALPAKA_ROOT_DIR}/cmake/dev.cmake")
    list(APPEND _DEFINITIONS_PRIVATE ${ALPAKA_DEV_COMPILE_OPTIONS})
endif()
if(ALPAKA_ACC_GPU_CUDA_ENABLE)
    list(APPEND _LINK_LIBRARIES_PRIVATE "general;${CUDA_CUDA_LIBRARY}")
    list(APPEND _DEFINITIONS_PRIVATE 
        "CUDA_API_PER_THREAD_DEFAULT_STREAM")
endif()


alpaka_add_executable(  
    ${_TARGET_NAME}
    src/main.cpp            src/Config.hpp
    src/Filecache.cpp       src/Filecache.hpp
    src/Devicefinder.cpp    src/Devicefinder.hpp
    src/Dispenser.cpp       src/Dispenser.hpp
    src/Bitmap.cpp          src/Bitmap.hpp
    src/Ringbuffer.hpp
    src/kernel/Calibration.hpp
    src/kernel/Correction.hpp
    src/kernel/Summation.hpp)

target_compile_options(
    ${_TARGET_NAME}
    PRIVATE ${_DEFINITIONS_PRIVATE}
    PRIVATE ${ALPAKA_DEV_COMPILE_OPTIONS})
target_include_directories(
    ${_TARGET_NAME}
    PRIVATE ${_INCLUDE_DIRECTORIES_PRIVATE}
    ~/Files/Libs/alpaka/include)
target_link_libraries(
    ${_TARGET_NAME}
    PUBLIC "alpaka"
    PRIVATE ${_LINK_LIBRARIES_PRIVATE})
