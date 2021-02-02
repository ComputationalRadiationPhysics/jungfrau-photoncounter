cmake_minimum_required(VERSION 3.12.0)
project("jungfrau-photoncounter")

# use CUDA_ROOT variable to find the CUDA root directory
cmake_policy(SET CMP0074 NEW)

set(ALPAKA_CXX_STANDARD "17" CACHE STRING "Force C++ version to 17. ")
set(ALPAKA_CUDA_NVCC_EXPT_RELAXED_CONSTEXPR OFF CACHE STRING "Turn expt-relaxed-contexpr off. ")

find_package(alpaka REQUIRED)

add_library(${CMAKE_PROJECT_NAME} INTERFACE)
target_include_directories(${CMAKE_PROJECT_NAME} INTERFACE include/${CMAKE_PROJECT_NAME})
target_link_libraries(${CMAKE_PROJECT_NAME} INTERFACE alpaka)

set(BENCHMARKING_ENABLED ON CACHE BOOL "Turn benchmarking on. ")
set(VERBOSE ON CACHE BOOL "Show debug/status messages. ")

# build main
alpaka_add_executable(photoncounter src/main.cpp)
target_include_directories(photoncounter PUBLIC include/ PUBLIC optional/include/tl/)
target_link_libraries(photoncounter PUBLIC alpaka PRIVATE ${CMAKE_PROJECT_NAME})
if(${VERBOSE})
  target_compile_definitions(photoncounter PRIVATE "-DVERBOSE")
endif()

if(${BENCHMARKING_ENABLED})
  # build benchmarks
  alpaka_add_executable(bench src/bench.cpp)
  target_include_directories(bench PUBLIC include/ PUBLIC optional/include/tl/)
  target_link_libraries(bench PUBLIC alpaka PRIVATE ${CMAKE_PROJECT_NAME})
  
  alpaka_add_executable(benchMultiple src/benchMultiple.cpp)
  target_include_directories(benchMultiple PUBLIC include/ PUBLIC optional/include/tl/)
  target_link_libraries(benchMultiple PUBLIC alpaka PRIVATE ${CMAKE_PROJECT_NAME})
endif()
