cmake_minimum_required(VERSION 3.11.0)
project("jungfrau-photoncounter")

set(ALPAKA_CXX_STANDARD "14" CACHE STRING "Force C++ version to 14. ")
set(ALPAKA_CUDA_NVCC_EXPT_RELAXED_CONSTEXPR OFF CACHE STRING "Turn expt-relaxed-contexpr off. ")

find_package(alpaka REQUIRED)

add_library(${CMAKE_PROJECT_NAME} INTERFACE)
target_include_directories(${CMAKE_PROJECT_NAME} INTERFACE include/${CMAKE_PROJECT_NAME})
target_link_libraries(${CMAKE_PROJECT_NAME} INTERFACE alpaka)

set(VERBOSE ON CACHE BOOL "Show debug/status messages. ")

# build benchmarks
alpaka_add_executable(bench src/bench.cpp)
target_include_directories(bench PUBLIC include/)
target_link_libraries(bench PUBLIC alpaka PRIVATE ${CMAKE_PROJECT_NAME})
if(${VERBOSE})
  target_compile_definitions(bench PRIVATE "-DVERBOSE")
endif()
