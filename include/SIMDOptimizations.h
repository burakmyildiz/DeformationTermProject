#pragma once

#include <Eigen/Core>
#include <vector>
#include <immintrin.h>

/**
 * @brief SIMD-optimized operations for physics simulation performance
 * Uses AVX2 (256-bit) instructions for maximum performance on modern CPUs
 */
class SIMDOptimizations {
public:
    /**
     * @brief SIMD-optimized batch computation of 9D extended positions
     * @param particles Vector of particles to update with extended position data
     * @note Processes 4 particles simultaneously using AVX2 for maximum throughput
     */
    static void batchUpdateExtendedPositions_AVX2(std::vector<class Particle>& particles);
    
    /**
     * @brief SIMD-optimized vector addition: result = a + b
     * @param a First input vector array
     * @param b Second input vector array
     * @param result Output array for element-wise sum
     * @param count Number of elements to process
     */
    static void vectorAdd_AVX2(const double* a, const double* b, double* result, size_t count);
    
    /**
     * @brief SIMD-optimized vector scaling: output = input * scale
     * @param input Input vector array
     * @param scale Scalar multiplication factor
     * @param output Output array for scaled values
     * @param count Number of elements to process
     */
    static void vectorScale_AVX2(const double* input, double scale, double* output, size_t count);
    
    /**
     * @brief SIMD-optimized element-wise vector multiplication: result = a * b
     * @param a First input vector array
     * @param b Second input vector array
     * @param result Output array for element-wise product
     * @param count Number of elements to process
     */
    static void vectorMultiply_AVX2(const double* a, const double* b, double* result, size_t count);
    
    /**
     * @brief SIMD-optimized distance calculations for collision detection
     * @param positions1 First set of 3D positions
     * @param count1 Number of positions in first set
     * @param positions2 Second set of 3D positions
     * @param count2 Number of positions in second set
     * @param distancesSq Output array for squared distances (size: count1 * count2)
     * @param minDistSq Output pointer for minimum squared distance found
     * @note Computes all pairwise squared distances between two position sets
     */
    static void computeDistanceSquared_AVX2(
        const Eigen::Vector3d* positions1, size_t count1,
        const Eigen::Vector3d* positions2, size_t count2,
        double* distancesSq, double* minDistSq);
    
    /**
     * @brief SIMD-optimized weighted centroid computation
     * @param positions Vector of 3D positions
     * @param weights Vector of weights for each position
     * @return Weighted centroid as 3D vector
     * @note Uses AVX2 for parallel accumulation of weighted sums
     */
    static Eigen::Vector3d computeCentroid_AVX2(
        const std::vector<Eigen::Vector3d>& positions,
        const std::vector<double>& weights);
    
    /**
     * @brief Checks if AVX2 instruction set is available at runtime
     * @return True if AVX2 is supported by the current CPU
     * @note Use this before calling AVX2-optimized functions for compatibility
     */
    static bool isAVX2Available();
    
private:
    /**
     * @brief Checks if memory pointer is properly aligned for SIMD operations
     * @param ptr Memory pointer to check
     * @param alignment Required alignment in bytes (default: 32 for AVX2)
     * @return True if pointer is aligned to specified boundary
     */
    static bool isAligned(const void* ptr, size_t alignment = 32);
    
    /**
     * @brief Computes largest count that maintains SIMD alignment
     * @param count Original element count
     * @param alignment Required element alignment (default: 4 for AVX2)
     * @return Aligned count that can be processed with SIMD
     */
    static size_t getAlignedCount(size_t count, size_t alignment = 4);
};