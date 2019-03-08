cmake_minimum_required(VERSION 3.11.0)
project("jungfrau-photoncounter")
find_package(alpaka REQUIRED)

alpaka_add_library(${CMAKE_PROJECT_NAME} src/Filecache.cpp)
target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC cxx_std_14)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include/${CMAKE_PROJECT_NAME})
target_include_directories(${CMAKE_PROJECT_NAME} INTERFACE include/)
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC alpaka)

alpaka_add_executable(photoncounter src/main.cpp)
target_compile_options(photoncounter PUBLIC cxx_std_14)
target_link_libraries(photoncounter PUBLIC alpaka PRIVATE ${CMAKE_PROJECT_NAME})
