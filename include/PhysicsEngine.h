#pragma once

#include <vector>
#include <memory>
#include <Eigen/Core>
#include "Particle.h"
#include "SimulationParameters.h"
#include "SpatialHashGrid.h"

class DeformableObject;

class PhysicsEngine {
private:
    SimulationParameters params;
    std::unique_ptr<SpatialHashGrid> spatialHashGrid;
    
public:
    /**
     * @brief Constructs physics engine with simulation parameters
     * @param p Simulation parameters (uses defaults if not provided)
     */
    PhysicsEngine(const SimulationParameters& p = SimulationParameters()) 
        : params(p), spatialHashGrid(std::make_unique<SpatialHashGrid>(p.spatialHashCellSize)) {}
    
    /**
     * @brief Updates all simulation parameters and recreates spatial hash grid
     * @param p New simulation parameters
     */
    void setParameters(const SimulationParameters& p) { 
        params = p; 
        spatialHashGrid = std::make_unique<SpatialHashGrid>(p.spatialHashCellSize);
    }
    
    /**
     * @brief Gets reference to current simulation parameters
     * @return Reference to internal parameters for modification
     */
    SimulationParameters& getParameters() { return params; }
    
    /**
     * @brief Applies gravitational acceleration to all particles
     * @param particles Vector of particles to affect
     * @note Uses equation: v = v + g*dt where g is gravitational acceleration
     */
    void applyGravity(std::vector<Particle>& particles);
    
    /**
     * @brief Applies velocity damping to simulate air resistance and energy loss
     * @param particles Vector of particles to damp
     * @note Multiplies velocity by damping factor: v *= damping
     */
    void applyDamping(std::vector<Particle>& particles);
    
    /**
     * @brief Handles collision between single particle and ground plane
     * @param particle Particle to check and resolve collision for
     * @note Applies position correction, velocity reflection, and friction
     */
    void handleGroundCollision(Particle& particle);
    
    /**
     * @brief Handles collision between single particle and room boundaries
     * @param particle Particle to check against walls, ceiling, and boundaries
     * @note Applies strong position correction and wall friction
     */
    void handleWallCollisions(Particle& particle);

    /**
     * @brief Applies shape matching velocity corrections from goal positions
     * @param particles Vector of particles to update
     * @param goalPositions Target positions from shape matching algorithm
     * @param dt Time step for velocity calculation
     * @note Implements Müller et al. equation: v = v + α*(g_i - x_i)/h
     */
    void applyShapeMatching(std::vector<Particle>& particles,
                           const std::vector<Eigen::Vector3d>& goalPositions,
                           double dt);
    
    /**
     * @brief Updates particle positions using Euler integration
     * @param particles Vector of particles to integrate
     * @param dt Time step duration
     * @note Performs x = x + v*dt with velocity capping for stability
     */
    void updatePositions(std::vector<Particle>& particles, double dt);
    
    /**
     * @brief Handles ground and wall collisions for all particles
     * @param particles Vector of particles to check for collisions
     * @note Batch processes all environmental collisions with optimization
     */
    void handleCollisions(std::vector<Particle>& particles);
    
    /**
     * @brief Legacy combined position update and collision handling
     * @param particles Vector of particles to update
     * @param dt Time step duration
     * @note Combines updatePositions() and handleCollisions() for compatibility
     */
    void updatePositionsAndCollisions(std::vector<Particle>& particles, double dt);
    
    /**
     * @brief Handles inter-object collision detection with adaptive method selection
     * @param objects Vector of deformable objects to check for collisions
     * @note Automatically selects brute force, spatial hash, or octree based on settings
     */
    void handleInterObjectCollisions(std::vector<std::unique_ptr<DeformableObject>>& objects);
    
    /**
     * @brief High-performance collision detection using spatial hash grid
     * @param objects Vector of deformable objects for collision detection
     * @note Optimized for uniform particle distributions with fixed cell sizes
     */
    void handleInterObjectCollisionsWithSpatialHashing(std::vector<std::unique_ptr<DeformableObject>>& objects);
    
    /**
     * @brief Hierarchical collision detection using adaptive octree structure
     * @param objects Vector of deformable objects for collision detection
     * @note Best for varying particle densities and dense collision scenarios
     */
    void handleInterObjectCollisionsWithOctree(std::vector<std::unique_ptr<DeformableObject>>& objects);
    
    /**
     * @brief Processes particle-particle collisions between two specific objects
     * @param obj1 First deformable object
     * @param obj2 Second deformable object  
     * @param relaxationFactor Collision response strength [0,1]
     * @note Applies position correction and velocity response with mass ratios
     */
    void processObjectPairCollisions(DeformableObject& obj1, DeformableObject& obj2, double relaxationFactor);
};
