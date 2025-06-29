#pragma once

#include <vector>
#include <array>
#include <string>
#include <Eigen/Core>
#include "Particle.h"
#include "ShapeMatcher.h"
#include "PhysicsEngine.h"

class DeformableObject {
private:
    std::vector<Particle> particles;
    std::vector<std::array<int, 3>> triangles;  
    ShapeMatcher shapeMatcher;
    
    
    std::string pointCloudName;
    std::string meshName;
    
    Eigen::Vector3d center;
    double scale;
    
    Eigen::Vector3d boundingCenter;
    double boundingRadius;
    bool boundingSphereNeedsUpdate;
    
    double adaptiveParticleRadius;
    
public:
    /**
     * @brief Constructs deformable object with unique identifier
     * @param name Unique name for Polyscope visualization (default: "deformable")
     */
    DeformableObject(const std::string& name = "deformable");
    
    /**
     * @brief Loads object geometry from OBJ file with scaling and positioning
     * @param filename Path to OBJ file containing mesh data
     * @param center 3D position to center the object (default: origin)
     * @param scale Uniform scaling factor for the mesh (default: 1.0)
     * @return True if loading succeeded, false on file error or parsing failure
     */
    bool loadFromOBJ(const std::string& filename, const Eigen::Vector3d& center = Eigen::Vector3d::Zero(), double scale = 1.0);
    
    /**
     * @brief Resets object to initial configuration
     * Restores all particles to rest positions and clears velocities
     */
    void reset();
    
    /**
     * @brief Performs one physics simulation step
     * @param engine Physics engine containing simulation parameters and methods
     * @param dt Time step duration for integration
     * @note Applies shape matching, integrates motion, handles collisions
     */
    void update(PhysicsEngine& engine, double dt);
    
    /**
     * @brief Initializes Polyscope visualization components
     * Creates point cloud and mesh representations for rendering
     */
    void initializeVisualization();
    
    /**
     * @brief Updates Polyscope visualization with current particle positions
     * Refreshes point cloud and mesh positions for current frame
     */
    void updateVisualization();
    
    /**
     * @brief Gets mutable reference to particle vector
     * @return Reference to internal particle data for modification
     */
    std::vector<Particle>& getParticles() { return particles; }
    
    /**
     * @brief Gets immutable reference to particle vector
     * @return Const reference to internal particle data for read-only access
     */
    const std::vector<Particle>& getParticles() const { return particles; }
    
    /**
     * @brief Updates bounding sphere for broad-phase collision culling
     * Recalculates center and radius from current particle positions
     */
    void updateBoundingSphere();
    
    /**
     * @brief Gets current bounding sphere center
     * @return 3D center position of the bounding sphere
     */
    const Eigen::Vector3d& getBoundingCenter() const { return boundingCenter; }
    
    /**
     * @brief Gets current bounding sphere radius
     * @return Radius of the bounding sphere
     */
    double getBoundingRadius() const { return boundingRadius; }
    
    /**
     * @brief Checks if bounding sphere needs recalculation
     * @return True if bounding sphere is outdated and needs updating
     */
    bool needsBoundingSphereUpdate() const { return boundingSphereNeedsUpdate; }
    
    /**
     * @brief Gets adaptive particle radius based on model density
     * @return Collision radius computed from particle distribution
     * @note Automatically scales with model size and particle count
     */
    double getAdaptiveParticleRadius() const { return adaptiveParticleRadius; }
    
    /**
     * @brief Applies external force to particles within radius of origin point
     * @param origin 3D center point for force application
     * @param force 3D force vector to apply
     * @param radius Distance within which particles are affected
     * @note Force falls off with distance from origin point
     */
    void applyForce(const Eigen::Vector3d& origin, const Eigen::Vector3d& force, double radius);
    
    /**
     * @brief Adjusts entire object position to prevent ground penetration
     * @param groundHeight Y-coordinate of the ground plane
     * @param particleRadius Radius of particles for collision detection
     * @note Lifts entire object uniformly to maintain shape integrity
     */
    void adjustForGroundContact(double groundHeight, double particleRadius);
};