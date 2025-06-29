#include "Particle.h"
#include "SIMDOptimizations.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructs a new particle with given position and mass
 * @param pos Initial 3D position vector
 * @param m Particle mass (default: 1.0)
 */
Particle::Particle(const Eigen::Vector3d& pos, double m) 
    : position(pos), velocity(Eigen::Vector3d::Zero()), restPosition(pos), mass(m) {
    // 1. Set up extended position data for shape matching
    updateExtendedPosition();
    updateExtendedRestPosition();
}

/**
 * @brief Resets particle to initial rest state
 * Restores position to rest position and clears velocity
 */
void Particle::reset() {
    // 1. Restore to initial state
    position = restPosition;
    velocity = Eigen::Vector3d::Zero();
    
    // 2. Update extended data to match reset position
    updateExtendedPosition();
}

/**
 * @brief Applies external force to particle using F=ma integration
 * @param force 3D force vector to apply
 * @param dt Time step duration for force integration
 * @note Updates velocity: v = v + (F/m)*dt
 */
void Particle::applyForce(const Eigen::Vector3d& force, double dt) {
    // 1. Convert force to acceleration and integrate into velocity
    velocity += (force / mass) * dt;
}

/**
 * @brief Performs Euler integration to update particle position
 * @param dt Time step duration for integration
 * @note Updates position: x = x + v*dt and refreshes extended position
 */
void Particle::integrate(double dt) {
    // 1. Move particle forward in time
    position += velocity * dt;
    
    // 2. Keep extended data synchronized
    updateExtendedPosition();
}

/**
 * @brief Computes kinetic energy of particle
 * @return Kinetic energy as 0.5 * m * v²
 */
double Particle::kineticEnergy() const {
    return 0.5 * mass * velocity.squaredNorm();
}

/**
 * @brief Updates 9D extended position representation from current 3D position
 * @note Called automatically after position changes for shape matching
 */
void Particle::updateExtendedPosition() {
    extendedPosition = to9D(position);
}

/**
 * @brief Updates 9D extended rest position representation from rest position
 * @note Called once during initialization for shape matching reference
 */
void Particle::updateExtendedRestPosition() {
    extendedRestPosition = to9D(restPosition);
}

/**
 * @brief Converts 3D position to 9D extended representation for quadratic deformations
 * @param pos 3D position vector [x,y,z]
 * @return 9D vector [x,y,z,x²,y²,z²,xy,yz,zx] for quadratic shape matching
 * @note Performance optimized with pre-computed squares and direct memory access
 */
Eigen::Matrix<double, 9, 1> Particle::to9D(const Eigen::Vector3d& pos) {
    // 1. Extract coordinates and compute squares efficiently
    const double x = pos.x(), y = pos.y(), z = pos.z();
    const double x2 = x * x, y2 = y * y, z2 = z * z;
    
    // 2. Build 9D vector with linear and quadratic terms
    Eigen::Matrix<double, 9, 1> extended;
    extended.data()[0] = x;
    extended.data()[1] = y;
    extended.data()[2] = z;
    extended.data()[3] = x2;
    extended.data()[4] = y2;
    extended.data()[5] = z2;
    extended.data()[6] = x * y;
    extended.data()[7] = y * z;
    extended.data()[8] = z * x;
                
    return extended;
}

/**
 * @brief High-performance batch update of extended positions using SIMD when available
 * @param particles Vector of particles to update extended positions for
 * @note Uses AVX2 SIMD optimization when available, falls back to scalar implementation
 */
void Particle::batchUpdateExtendedPositions(std::vector<Particle>& particles) {
    // 1. Use SIMD acceleration when possible
    if (SIMDOptimizations::isAVX2Available() && particles.size() >= 4) {
        SIMDOptimizations::batchUpdateExtendedPositions_AVX2(particles);
    } else {
        // 2. Process particles one by one with optimized scalar code
        #ifdef USE_OPENMP
        #pragma omp parallel for
        #endif
        for (size_t i = 0; i < particles.size(); ++i) {
            auto& particle = particles[i];
            const double x = particle.position.x();
            const double y = particle.position.y(); 
            const double z = particle.position.z();
            const double x2 = x * x, y2 = y * y, z2 = z * z;
            
            double* data = particle.extendedPosition.data();
            data[0] = x;
            data[1] = y;
            data[2] = z;
            data[3] = x2;
            data[4] = y2;
            data[5] = z2;
            data[6] = x * y;
            data[7] = y * z;
            data[8] = z * x;
        }
    }
}