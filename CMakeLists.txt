cmake_minimum_required(VERSION 3.11.0)
project("jungfrau-photoncounter")
find_package(alpaka REQUIRED)

add_library(${CMAKE_PROJECT_NAME} INTERFACE)
target_include_directories(${CMAKE_PROJECT_NAME} INTERFACE include/${CMAKE_PROJECT_NAME})
target_link_libraries(${CMAKE_PROJECT_NAME} INTERFACE alpaka)

#add_subdirectory(${PROJECT_SOURCE_DIR}/bench/)

alpaka_add_executable(photoncounter src/main.cpp)
target_include_directories(photoncounter PUBLIC include/)
target_link_libraries(photoncounter PUBLIC alpaka PRIVATE ${CMAKE_PROJECT_NAME})
