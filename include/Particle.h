#pragma once

#include <Eigen/Core>

class Particle {
public:
    Eigen::Vector3d position;      // Current position x_i
    Eigen::Vector3d velocity;      // Current velocity v_i
    Eigen::Vector3d restPosition;  // Original position
    double mass;                   // Particle mass m_i
    
    Eigen::Matrix<double, 9, 1> extendedPosition;     // [x, y, z, x², y², z², xy, yz, zx]
    Eigen::Matrix<double, 9, 1> extendedRestPosition; // [x, y, z, x², y², z², xy, yz, zx] at rest
    
    /**
     * @brief Constructs a new particle with given position and mass
     * @param pos Initial 3D position vector
     * @param m Particle mass (default: 1.0)
     */
    Particle(const Eigen::Vector3d& pos, double m = 1.0);
    
    /**
     * @brief Resets particle to its initial rest state
     * Restores position to restPosition, clears velocity, updates extended representations
     */
    void reset();
    
    /**
     * @brief Applies external force to particle for one timestep using F = ma
     * @param force 3D force vector to apply
     * @param dt Time step duration
     */
    void applyForce(const Eigen::Vector3d& force, double dt);
    
    /**
     * @brief Performs simple Euler integration: position += velocity * dt
     * @param dt Time step duration
     * @note Does not include shape matching forces - used for basic physics updates
     */
    void integrate(double dt);
    
    /**
     * @brief Calculates particle's kinetic energy: KE = 0.5 * mass * velocity²
     * @return Kinetic energy value
     */
    double kineticEnergy() const;
    
    /**
     * @brief Updates 9D extended position from current 3D position
     * Computes [x, y, z, x², y², z², xy, yz, zx] for quadratic deformations
     */
    void updateExtendedPosition();
    
    /**
     * @brief Updates 9D extended rest position from 3D rest position
     * Computes rest state extended representation for shape matching
     */
    void updateExtendedRestPosition();
    
    /**
     * @brief Converts 3D position vector to 9D extended representation
     * @param pos 3D position vector
     * @return 9D vector [x, y, z, x², y², z², xy, yz, zx]
     */
    static Eigen::Matrix<double, 9, 1> to9D(const Eigen::Vector3d& pos);
    
    /**
     * @brief Batch updates extended positions for multiple particles to reduce overhead
     * @param particles Vector of particles to update
     * @note Performance optimization: eliminates function call overhead for N particles per frame
     */
    static void batchUpdateExtendedPositions(std::vector<Particle>& particles);
};