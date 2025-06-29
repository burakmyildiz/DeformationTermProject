#include "SIMDOptimizations.h"
#include "Particle.h"
#include <algorithm>
#include <cstring>

/**
 * @brief Checks if AVX2 instruction set is available at runtime
 * @return True if AVX2 is supported by the current CPU
 * @note Use this before calling AVX2-optimized functions for compatibility
 */
bool SIMDOptimizations::isAVX2Available() {
    // 1. Ask the CPU what features it supports
    unsigned int eax, ebx, ecx, edx;
    __asm__ __volatile__("cpuid": "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                        : "a"(7), "c"(0));
    return (ebx & (1 << 5)) != 0;
}

/**
 * @brief Checks if memory pointer is properly aligned for SIMD operations
 * @param ptr Memory pointer to check
 * @param alignment Required alignment in bytes (default: 32 for AVX2)
 * @return True if pointer is aligned to specified boundary
 */
bool SIMDOptimizations::isAligned(const void* ptr, size_t alignment) {
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

/**
 * @brief Computes largest count that maintains SIMD alignment
 * @param count Original element count
 * @param alignment Required element alignment (default: 4 for AVX2)
 * @return Aligned count that can be processed with SIMD
 */
size_t SIMDOptimizations::getAlignedCount(size_t count, size_t alignment) {
    return (count / alignment) * alignment;
}

/**
 * @brief SIMD-optimized batch computation of 9D extended positions
 * @param particles Vector of particles to update with extended position data
 * @note Processes 4 particles simultaneously using AVX2 for maximum throughput
 */
void SIMDOptimizations::batchUpdateExtendedPositions_AVX2(std::vector<Particle>& particles) {
    if (!isAVX2Available() || particles.empty()) {
        // 1. Use regular method if fast processing unavailable
        Particle::batchUpdateExtendedPositions(particles);
        return;
    }
    
    const size_t count = particles.size();
    const size_t alignedCount = getAlignedCount(count, 4);
    
    // 2. Process groups of 4 particles simultaneously
    for (size_t i = 0; i < alignedCount; i += 4) {
        // 3. Load x coordinates from 4 particles
        __m256d x_vec = _mm256_set_pd(
            particles[i + 3].position.x(),
            particles[i + 2].position.x(),
            particles[i + 1].position.x(),
            particles[i + 0].position.x()
        );
        
        // 4. Load y coordinates from 4 particles
        __m256d y_vec = _mm256_set_pd(
            particles[i + 3].position.y(),
            particles[i + 2].position.y(),
            particles[i + 1].position.y(),
            particles[i + 0].position.y()
        );
        
        // 5. Load z coordinates from 4 particles
        __m256d z_vec = _mm256_set_pd(
            particles[i + 3].position.z(),
            particles[i + 2].position.z(),
            particles[i + 1].position.z(),
            particles[i + 0].position.z()
        );
        
        // 6. Calculate squares for all 4 particles at once
        __m256d x2_vec = _mm256_mul_pd(x_vec, x_vec);
        __m256d y2_vec = _mm256_mul_pd(y_vec, y_vec);
        __m256d z2_vec = _mm256_mul_pd(z_vec, z_vec);
        
        // 7. Calculate cross products for all 4 particles
        __m256d xy_vec = _mm256_mul_pd(x_vec, y_vec);
        __m256d yz_vec = _mm256_mul_pd(y_vec, z_vec);
        __m256d zx_vec = _mm256_mul_pd(z_vec, x_vec);
        
        // 8. Extract computed values from SIMD registers
        alignas(32) double x_vals[4], y_vals[4], z_vals[4];
        alignas(32) double x2_vals[4], y2_vals[4], z2_vals[4];
        alignas(32) double xy_vals[4], yz_vals[4], zx_vals[4];
        
        _mm256_store_pd(x_vals, x_vec);
        _mm256_store_pd(y_vals, y_vec);
        _mm256_store_pd(z_vals, z_vec);
        _mm256_store_pd(x2_vals, x2_vec);
        _mm256_store_pd(y2_vals, y2_vec);
        _mm256_store_pd(z2_vals, z2_vec);
        _mm256_store_pd(xy_vals, xy_vec);
        _mm256_store_pd(yz_vals, yz_vec);
        _mm256_store_pd(zx_vals, zx_vec);
        
        // 9. Store extended position data for each particle
        for (int j = 0; j < 4; ++j) {
            double* data = particles[i + j].extendedPosition.data();
            data[0] = x_vals[j];
            data[1] = y_vals[j];
            data[2] = z_vals[j];
            data[3] = x2_vals[j];
            data[4] = y2_vals[j];
            data[5] = z2_vals[j];
            data[6] = xy_vals[j];
            data[7] = yz_vals[j];
            data[8] = zx_vals[j];
        }
    }
    
    // 10. Handle leftover particles individually
    for (size_t i = alignedCount; i < count; ++i) {
        const double x = particles[i].position.x();
        const double y = particles[i].position.y();
        const double z = particles[i].position.z();
        
        const double x2 = x * x, y2 = y * y, z2 = z * z;
        
        double* data = particles[i].extendedPosition.data();
        data[0] = x; data[1] = y; data[2] = z;
        data[3] = x2; data[4] = y2; data[5] = z2;
        data[6] = x * y; data[7] = y * z; data[8] = z * x;
    }
}

/**
 * @brief SIMD-optimized vector addition: result = a + b
 * @param a First input vector array
 * @param b Second input vector array
 * @param result Output array for element-wise sum
 * @param count Number of elements to process
 */
void SIMDOptimizations::vectorAdd_AVX2(const double* a, const double* b, double* result, size_t count) {
    if (!isAVX2Available()) return;
    
    const size_t alignedCount = getAlignedCount(count, 4);
    
    // 1. Add 4 numbers at once using SIMD
    for (size_t i = 0; i < alignedCount; i += 4) {
        __m256d a_vec = _mm256_load_pd(&a[i]);
        __m256d b_vec = _mm256_load_pd(&b[i]);
        __m256d result_vec = _mm256_add_pd(a_vec, b_vec);
        _mm256_store_pd(&result[i], result_vec);
    }
    
    // 2. Handle remaining numbers individually
    for (size_t i = alignedCount; i < count; ++i) {
        result[i] = a[i] + b[i];
    }
}

/**
 * @brief SIMD-optimized vector scaling: output = input * scale
 * @param input Input vector array
 * @param scale Scalar multiplication factor
 * @param output Output array for scaled values
 * @param count Number of elements to process
 */
void SIMDOptimizations::vectorScale_AVX2(const double* input, double scale, double* output, size_t count) {
    if (!isAVX2Available()) return;
    
    const size_t alignedCount = getAlignedCount(count, 4);
    __m256d scale_vec = _mm256_set1_pd(scale);
    
    // 1. Multiply 4 numbers by scale factor simultaneously
    for (size_t i = 0; i < alignedCount; i += 4) {
        __m256d input_vec = _mm256_load_pd(&input[i]);
        __m256d result_vec = _mm256_mul_pd(input_vec, scale_vec);
        _mm256_store_pd(&output[i], result_vec);
    }
    
    // 2. Handle remaining numbers individually
    for (size_t i = alignedCount; i < count; ++i) {
        output[i] = input[i] * scale;
    }
}

/**
 * @brief SIMD-optimized element-wise vector multiplication: result = a * b
 * @param a First input vector array
 * @param b Second input vector array
 * @param result Output array for element-wise product
 * @param count Number of elements to process
 */
void SIMDOptimizations::vectorMultiply_AVX2(const double* a, const double* b, double* result, size_t count) {
    if (!isAVX2Available()) return;
    
    const size_t alignedCount = getAlignedCount(count, 4);
    
    // 1. Multiply 4 pairs of numbers simultaneously
    for (size_t i = 0; i < alignedCount; i += 4) {
        __m256d a_vec = _mm256_load_pd(&a[i]);
        __m256d b_vec = _mm256_load_pd(&b[i]);
        __m256d result_vec = _mm256_mul_pd(a_vec, b_vec);
        _mm256_store_pd(&result[i], result_vec);
    }
    
    // 2. Handle remaining pairs individually
    for (size_t i = alignedCount; i < count; ++i) {
        result[i] = a[i] * b[i];
    }
}

/**
 * @brief SIMD-optimized weighted centroid computation
 * @param positions Vector of 3D positions
 * @param weights Vector of weights for each position
 * @return Weighted centroid as 3D vector
 * @note Uses AVX2 for parallel accumulation of weighted sums
 */
Eigen::Vector3d SIMDOptimizations::computeCentroid_AVX2(
    const std::vector<Eigen::Vector3d>& positions,
    const std::vector<double>& weights) {
    
    if (!isAVX2Available() || positions.size() != weights.size() || positions.empty()) {
        // 1. Use standard method if fast processing unavailable
        Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
        double totalWeight = 0.0;
        for (size_t i = 0; i < positions.size(); ++i) {
            centroid += weights[i] * positions[i];
            totalWeight += weights[i];
        }
        return centroid / totalWeight;
    }
    
    const size_t count = positions.size();
    const size_t alignedCount = getAlignedCount(count, 4);
    
    // 2. Set up running totals for each coordinate
    __m256d sum_x = _mm256_setzero_pd();
    __m256d sum_y = _mm256_setzero_pd();
    __m256d sum_z = _mm256_setzero_pd();
    __m256d sum_w = _mm256_setzero_pd();
    
    // 3. Process 4 positions simultaneously
    for (size_t i = 0; i < alignedCount; i += 4) {
        // 4. Load weight values for 4 particles
        __m256d w_vec = _mm256_set_pd(weights[i+3], weights[i+2], weights[i+1], weights[i]);
        
        // 5. Load position coordinates for 4 particles
        __m256d x_vec = _mm256_set_pd(
            positions[i+3].x(), positions[i+2].x(), 
            positions[i+1].x(), positions[i].x()
        );
        __m256d y_vec = _mm256_set_pd(
            positions[i+3].y(), positions[i+2].y(), 
            positions[i+1].y(), positions[i].y()
        );
        __m256d z_vec = _mm256_set_pd(
            positions[i+3].z(), positions[i+2].z(), 
            positions[i+1].z(), positions[i].z()
        );
        
        // 6. Add weighted contributions to running totals
        sum_x = _mm256_fmadd_pd(x_vec, w_vec, sum_x);
        sum_y = _mm256_fmadd_pd(y_vec, w_vec, sum_y);
        sum_z = _mm256_fmadd_pd(z_vec, w_vec, sum_z);
        sum_w = _mm256_add_pd(sum_w, w_vec);
    }
    
    // 7. Extract and combine the 4 accumulated values
    alignas(32) double x_vals[4], y_vals[4], z_vals[4], w_vals[4];
    _mm256_store_pd(x_vals, sum_x);
    _mm256_store_pd(y_vals, sum_y);
    _mm256_store_pd(z_vals, sum_z);
    _mm256_store_pd(w_vals, sum_w);
    
    double totalX = x_vals[0] + x_vals[1] + x_vals[2] + x_vals[3];
    double totalY = y_vals[0] + y_vals[1] + y_vals[2] + y_vals[3];
    double totalZ = z_vals[0] + z_vals[1] + z_vals[2] + z_vals[3];
    double totalW = w_vals[0] + w_vals[1] + w_vals[2] + w_vals[3];
    
    // 8. Process remaining positions individually
    for (size_t i = alignedCount; i < count; ++i) {
        totalX += positions[i].x() * weights[i];
        totalY += positions[i].y() * weights[i];
        totalZ += positions[i].z() * weights[i];
        totalW += weights[i];
    }
    
    return Eigen::Vector3d(totalX / totalW, totalY / totalW, totalZ / totalW);
}