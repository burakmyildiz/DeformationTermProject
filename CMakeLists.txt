cmake_minimum_required(VERSION 3.10)
project(DeformationSimulation)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

add_subdirectory(polyscope)

add_executable(deformation_sim
    src/main.cpp
    src/Application.cpp
    src/Particle.cpp
    src/ShapeMatcher.cpp
    src/PhysicsEngine.cpp
    src/DeformableObject.cpp
    src/SIMDOptimizations.cpp
    src/SpatialHashGrid.cpp
)

target_include_directories(deformation_sim PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(deformation_sim 
    Eigen3::Eigen
    polyscope
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(deformation_sim OpenMP::OpenMP_CXX)
    target_compile_definitions(deformation_sim PRIVATE USE_OPENMP)
    message(STATUS "OpenMP found and enabled")
else()
    message(STATUS "OpenMP not found - parallel features disabled")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mfma")

# SIMD optimizations: Enable AVX2 and FMA for maximum performance
target_compile_options(deformation_sim PRIVATE
    $<$<CONFIG:Release>:-mavx2>
    $<$<CONFIG:Release>:-mfma>
    $<$<CONFIG:Release>:-march=native> 
)