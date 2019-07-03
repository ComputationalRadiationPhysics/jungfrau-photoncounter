cmake_minimum_required(VERSION 3.11.0)
project("jungfrau-photoncounter")
find_package(alpaka REQUIRED)

add_library(${CMAKE_PROJECT_NAME} INTERFACE)
target_include_directories(${CMAKE_PROJECT_NAME} INTERFACE include/${CMAKE_PROJECT_NAME})
target_link_libraries(${CMAKE_PROJECT_NAME} INTERFACE alpaka)

set(BENCHMARKING_ENABLED OFF CACHE BOOL "Turn benchmarking on. ")
set(VERBOSE ON CACHE BOOL "Show debug/status messages. ")

# build main
alpaka_add_executable(photoncounter src/main.cpp)
target_include_directories(photoncounter PUBLIC include/)
target_link_libraries(photoncounter PUBLIC alpaka PRIVATE ${CMAKE_PROJECT_NAME})
if(${VERBOSE})
  target_compile_definitions(photoncounter PRIVATE "-DVERBOSE")
endif()

if(${BENCHMARKING_ENABLED})
  # build benchmarks
  add_subdirectory(benchmark/)
  alpaka_add_executable(bench src/bench.cpp)
  target_include_directories(bench PUBLIC include/)
  target_link_libraries(bench PUBLIC alpaka benchmark PRIVATE ${CMAKE_PROJECT_NAME})
endif()
