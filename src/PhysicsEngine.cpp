#include "PhysicsEngine.h"
#include "DeformableObject.h"
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <tuple>

#include "OctreeCollisionDetection.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

/**
 * @brief Applies gravitational acceleration to all particles
 * @param particles Vector of particles to affect
 * @note Uses equation: v = v + g*dt where g is gravitational acceleration
 */
void PhysicsEngine::applyGravity(std::vector<Particle>& particles) {
    // 1. Set up gravity force acting downward
    Eigen::Vector3d gravityAcceleration(0, params.gravity, 0);
    
    // 2. Apply gravity to each particle's velocity
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].velocity += gravityAcceleration * params.timeStep;
    }
}

/**
 * @brief Applies velocity damping to simulate air resistance and energy loss
 * @param particles Vector of particles to damp
 * @note Multiplies velocity by damping factor: v *= damping
 */
void PhysicsEngine::applyDamping(std::vector<Particle>& particles) {
    // 1. Reduce velocity to simulate air resistance
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < particles.size(); ++i) {
        particles[i].velocity *= params.damping;
    }
}

/**
 * @brief Handles collision between single particle and ground plane
 * @param particle Particle to check and resolve collision for
 * @note Applies position correction, velocity reflection, and friction
 */
void PhysicsEngine::handleGroundCollision(Particle& particle) {
    double groundLevel = params.groundHeight + params.particleRadius;
    
    if (particle.position.y() < groundLevel) {
        // 1. ALWAYS push particle completely back to ground level
        particle.position.y() = groundLevel;
        
        // 2. Handle velocity based on restitution
        if (particle.velocity.y() < 0) {
            if (params.restitution < 0.01f) {
                particle.velocity.y() = 0.0;
                particle.velocity.x() *= params.friction * params.groundStickiness;
                particle.velocity.z() *= params.friction * params.groundStickiness;
            } else {
                particle.velocity.y() = -particle.velocity.y() * params.restitution;
                particle.velocity.x() *= params.friction;
                particle.velocity.z() *= params.friction;
            }
        }
    }
}

/**
 * @brief Handles collision between single particle and room boundaries
 * @param particle Particle to check against walls, ceiling, and boundaries
 * @note Applies strong position correction and wall friction
 */
void PhysicsEngine::handleWallCollisions(Particle& particle) {
    bool collisionOccurred = false;
    
    // 1. Check left wall collision
    double leftWall = -params.roomSize + params.particleRadius;
    if (particle.position.x() < leftWall) {
        particle.position.x() = leftWall;
        if (particle.velocity.x() < 0) {
            particle.velocity.x() = -particle.velocity.x() * params.restitution;
        }
        collisionOccurred = true;
    }
    
    // 2. Check right wall collision
    double rightWall = params.roomSize - params.particleRadius;
    if (particle.position.x() > rightWall) {
        particle.position.x() = rightWall;
        if (particle.velocity.x() > 0) {
            particle.velocity.x() = -particle.velocity.x() * params.restitution;
        }
        collisionOccurred = true;
    }
    
    // 3. Check back wall collision
    double backWall = -params.roomSize + params.particleRadius;
    if (particle.position.z() < backWall) {
        particle.position.z() = backWall;
        if (particle.velocity.z() < 0) {
            particle.velocity.z() = -particle.velocity.z() * params.restitution;
        }
        collisionOccurred = true;
    }
    
    // 4. Check front wall collision
    double frontWall = params.roomSize - params.particleRadius;
    if (particle.position.z() > frontWall) {
        particle.position.z() = frontWall;
        if (particle.velocity.z() > 0) {
            particle.velocity.z() = -particle.velocity.z() * params.restitution;
        }
        collisionOccurred = true;
    }
    
    // 5. Check ceiling collision
    double ceiling = params.groundHeight + params.roomHeight - params.particleRadius;
    if (particle.position.y() > ceiling) {
        particle.position.y() = ceiling;
        if (particle.velocity.y() > 0) {
            particle.velocity.y() = -particle.velocity.y() * params.restitution;
        }
        collisionOccurred = true;
    }
    
    // 6. Apply wall friction if any collision happened
    if (collisionOccurred) {
        particle.velocity *= params.wallFriction;
    }
}

/**
 * @brief Applies shape matching velocity corrections from goal positions
 * @param particles Vector of particles to update
 * @param goalPositions Target positions from shape matching algorithm
 * @param dt Time step for velocity calculation
 * @note Implements Müller et al. equation: v = v + α*(g_i - x_i)/h
 */
void PhysicsEngine::applyShapeMatching(std::vector<Particle>& particles,
                                       const std::vector<Eigen::Vector3d>& goalPositions,
                                       double dt) {
    // 1. Calculate how strong the shape matching should be
    float adaptiveAlpha = std::min(static_cast<float>(dt) / params.adaptiveStiffnessTau, 1.0f);
    
    // 2. Push each particle toward its goal position
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& p = particles[i];
        
        // 3. Apply shape matching force uniformly - don't suppress near ground
        Eigen::Vector3d correction = adaptiveAlpha * (goalPositions[i] - p.position) / dt;
        p.velocity += correction;
    }
}

/**
 * @brief Updates particle positions using Euler integration
 * @param particles Vector of particles to integrate
 * @param dt Time step duration
 * @note Performs x = x + v*dt with velocity capping for stability
 */
void PhysicsEngine::updatePositions(std::vector<Particle>& particles, double dt) {
    // 1. Move each particle based on its velocity
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& p = particles[i];
        // 2. Prevent velocities from getting too high
        if (p.velocity.norm() > params.maxVelocity) {
            p.velocity = p.velocity.normalized() * params.maxVelocity;
        }
        
        // 3. Update position using simple physics
        p.position += p.velocity * dt;
    }
    
    // 4. Update extended position data efficiently
    Particle::batchUpdateExtendedPositions(particles);
}

/**
 * @brief Handles ground and wall collisions for all particles
 * @param particles Vector of particles to check for collisions
 * @note Batch processes all environmental collisions with optimization
 */
void PhysicsEngine::handleCollisions(std::vector<Particle>& particles) {
    bool anyCollisionOccurred = false;
    
    // 1. Check each particle for collisions with environment
    #ifdef USE_OPENMP
    #pragma omp parallel for reduction(||:anyCollisionOccurred)
    #endif
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& p = particles[i];
        // 2. Handle ground collisions
        if (p.position.y() < params.groundHeight + params.particleRadius) {
            handleGroundCollision(p);
            anyCollisionOccurred = true;
        }
        
        // 3. Handle wall collisions
        handleWallCollisions(p);
    }
    
    // 4. Update extended positions only if needed for performance
    if (anyCollisionOccurred) {
        Particle::batchUpdateExtendedPositions(particles);
    }
}

/**
 * @brief Legacy combined position update and collision handling
 * @param particles Vector of particles to update
 * @param dt Time step duration
 * @note Combines updatePositions() and handleCollisions() for compatibility
 */
void PhysicsEngine::updatePositionsAndCollisions(std::vector<Particle>& particles, double dt) {
    // 1. Move particles forward in time
    updatePositions(particles, dt);
    // 2. Handle any collisions that occurred
    handleCollisions(particles);
}

/**
 * @brief Handles inter-object collision detection with adaptive method selection
 * @param objects Vector of deformable objects to check for collisions
 * @note Automatically selects brute force, spatial hash, or octree based on settings
 */
void PhysicsEngine::handleInterObjectCollisions(
    std::vector<std::unique_ptr<DeformableObject>>& objects) {
    
    // 1. Count total particles to adjust performance
    size_t totalParticles = 0;
    for (const auto& obj : objects) {
        if (obj) totalParticles += obj->getParticles().size();
    }
    
    // 2. Use fewer iterations for large simulations
    int numIterations = 4;  
    if (params.useAdaptiveIterations) {
        if (totalParticles > 10000) numIterations = 1;
        else if (totalParticles > 5000) numIterations = 2;
        else if (totalParticles > 2000) numIterations = 3;
        else if (totalParticles > 1000) numIterations = 3;  
        else if (totalParticles > 500) numIterations = 4;   
    }
    
    const double relaxationFactor = 1.0; 

    // 3. Use faster collision detection methods if available
    if (objects.size() > 1) {
        if (params.useOctree) {
            handleInterObjectCollisionsWithOctree(objects);
            return;
        } else if (params.useSpatialHashing) {
            handleInterObjectCollisionsWithSpatialHashing(objects);
            return;
        }
    }

    // 4. Use basic method when no optimization is enabled
    for (int iter = 0; iter < numIterations; ++iter) {
        for (size_t i = 0; i < objects.size(); ++i) {
            for (size_t j = i + 1; j < objects.size(); ++j) {
                if (!objects[i] || !objects[j]) continue;
                
                // 5. Update object boundaries for quick distance checks
                if (objects[i]->needsBoundingSphereUpdate()) {
                    objects[i]->updateBoundingSphere();
                }
                if (objects[j]->needsBoundingSphereUpdate()) {
                    objects[j]->updateBoundingSphere();
                }
                
                // 6. Check if objects are close enough to possibly collide
                Eigen::Vector3d centerDiff = objects[j]->getBoundingCenter() - objects[i]->getBoundingCenter();
                double centerDistance = centerDiff.norm();
                double combinedRadius = objects[i]->getBoundingRadius() + objects[j]->getBoundingRadius();
                
                // 7. Skip detailed checking if objects are far apart
                if (centerDistance > combinedRadius) {
                    continue;
                }
                
                processObjectPairCollisions(*objects[i], *objects[j], relaxationFactor);
            }
        }
    }
    
    // 8. Update extended position data for all objects
    for (auto& object : objects) {
        if (object) {
            Particle::batchUpdateExtendedPositions(object->getParticles());
        }
    }
}

/**
 * @brief Processes particle-particle collisions between two specific objects
 * @param obj1 First deformable object
 * @param obj2 Second deformable object  
 * @param relaxationFactor Collision response strength [0,1]
 * @note Applies position correction and velocity response with mass ratios
 */
void PhysicsEngine::processObjectPairCollisions(DeformableObject& obj1, DeformableObject& obj2, double relaxationFactor) {
    auto& particles1 = obj1.getParticles();
    auto& particles2 = obj2.getParticles();

    // 1. Get collision radii for both objects
    double adaptiveRadius1 = obj1.getAdaptiveParticleRadius();
    double adaptiveRadius2 = obj2.getAdaptiveParticleRadius();
    double adaptiveMinDist = (adaptiveRadius1 + adaptiveRadius2) * params.contactOffset;
    
    // 2. Calculate squared distance for faster comparison
    const double minDistSq = adaptiveMinDist * adaptiveMinDist;
    
    // 3. Check every particle from first object against second object
    for (auto& p1 : particles1) {
        for (auto& p2 : particles2) {
            Eigen::Vector3d diff = p2.position - p1.position;
            
            // 4. Use squared distance to avoid expensive square root calculation
            double distSq = diff.squaredNorm();
            
            if (distSq < minDistSq && distSq > 1e-12) {
                // 5. Calculate collision details only when needed
                double dist = std::sqrt(distSq);
                double invDist = 1.0 / dist;
                Eigen::Vector3d normal = diff * invDist;
                double penetration = adaptiveMinDist - dist;
                
                // 6. Calculate mass ratios for realistic collision response
                double totalMass = p1.mass + p2.mass;
                double invTotalMass = 1.0 / totalMass;
                double mass1Ratio = p2.mass * invTotalMass;
                double mass2Ratio = p1.mass * invTotalMass;
                
                // 7. Separate overlapping particles
                double correctionMagnitude = std::min(
                    penetration * params.separationStrength * relaxationFactor,
                    static_cast<double>(params.maxPositionCorrection)
                );
                
                Eigen::Vector3d correction = normal * correctionMagnitude;
                p1.position -= correction * mass1Ratio;
                p2.position += correction * mass2Ratio;

                // 8. Adjust velocities if particles are moving toward each other
                double relVel = (p2.velocity - p1.velocity).dot(normal);
                if (relVel < 0) {
                    double impulse = -relVel * (1.0 + params.restitution) * relaxationFactor;
                    Eigen::Vector3d velocityChange = normal * impulse;
                    p1.velocity -= velocityChange * mass1Ratio;
                    p2.velocity += velocityChange * mass2Ratio;
                }
            }
        }
    }
}

/**
 * @brief High-performance collision detection using spatial hash grid
 * @param objects Vector of deformable objects for collision detection
 * @note Optimized for uniform particle distributions with fixed cell sizes
 */
void PhysicsEngine::handleInterObjectCollisionsWithSpatialHashing(
    std::vector<std::unique_ptr<DeformableObject>>& objects) {
    
    // 1. Count total particles to adjust performance
    size_t totalParticles = 0;
    for (const auto& obj : objects) {
        if (obj) totalParticles += obj->getParticles().size();
    }
    
    // 2. Use fewer iterations for large simulations
    int numIterations = 4;  // Reduced default for better performance
    if (params.useAdaptiveIterations) {
        if (totalParticles > 10000) numIterations = 1;
        else if (totalParticles > 5000) numIterations = 2;
        else if (totalParticles > 2000) numIterations = 3;
        else if (totalParticles > 1000) numIterations = 3;  // Reduced from 4
        else if (totalParticles > 500) numIterations = 4;   // Added middle tier
    }
    
    const double relaxationFactor = 1.0;
    
    // 3. Skip if no collisions possible
    if (objects.size() <= 1) return;
    
    // 4. Set up tracking for particle information
    struct ParticleInfo {
        size_t objectIndex;
        size_t localIndex;
        double radius;
    };
    std::vector<ParticleInfo> particleMap;
    
    // 5. Use simple method for small simulations
    if (totalParticles < 100) {
        handleInterObjectCollisions(objects);
        return;
    }
    
    // 6. Update object boundaries first
    std::vector<std::pair<size_t, size_t>> collidingObjectPairs;
    for (size_t i = 0; i < objects.size(); ++i) {
        if (!objects[i]) continue;
        if (objects[i]->needsBoundingSphereUpdate()) {
            objects[i]->updateBoundingSphere();
        }
    }
    
    // 7. Find which objects are close enough to collide
    for (size_t i = 0; i < objects.size(); ++i) {
        if (!objects[i]) continue;
        for (size_t j = i + 1; j < objects.size(); ++j) {
            if (!objects[j]) continue;
            
            Eigen::Vector3d centerDiff = objects[j]->getBoundingCenter() - objects[i]->getBoundingCenter();
            double centerDistance = centerDiff.norm();
            double combinedRadius = objects[i]->getBoundingRadius() + objects[j]->getBoundingRadius();
            
            // Add extra space for particle sizes
            combinedRadius += 2.0 * std::max(objects[i]->getAdaptiveParticleRadius(), 
                                            objects[j]->getAdaptiveParticleRadius());
            
            if (centerDistance <= combinedRadius) {
                collidingObjectPairs.push_back({i, j});
            }
        }
    }
    
    // 8. Skip if no objects are close
    if (collidingObjectPairs.empty()) return;
    
    // 9. Create spatial grid for fast collision detection
    SpatialHashGrid globalGrid(params.spatialHashCellSize);
    
    // 10. Only track objects that might actually collide
    std::unordered_set<size_t> activeObjects;
    for (const auto& [i, j] : collidingObjectPairs) {
        activeObjects.insert(i);
        activeObjects.insert(j);
    }
    
    size_t globalParticleId = 0;
    for (size_t objIdx : activeObjects) {
        auto& particles = objects[objIdx]->getParticles();
        double radius = objects[objIdx]->getAdaptiveParticleRadius();
        
        // 11. Store information for each particle
        for (size_t localIdx = 0; localIdx < particles.size(); ++localIdx) {
            particleMap.push_back({objIdx, localIdx, radius});
            globalParticleId++;
        }
    }
    
    // 12. Perform multiple collision resolution passes
    for (int iter = 0; iter < numIterations; ++iter) {
        // 13. Reset grid and add particles at their current positions
        globalGrid.clear();
        globalParticleId = 0;
        for (size_t objIdx : activeObjects) {
            auto& particles = objects[objIdx]->getParticles();
            for (size_t localIdx = 0; localIdx < particles.size(); ++localIdx) {
                globalGrid.insertParticleWithObjectId(globalParticleId, objIdx, particles[localIdx].position);
                globalParticleId++;
            }
        }
        
        // 14. Find the largest particle radius
        double maxRadius = 0.0;
        for (const auto& obj : objects) {
            if (obj) {
                maxRadius = std::max(maxRadius, obj->getAdaptiveParticleRadius());
            }
        }
        
        // 15. Get potential collision pairs from grid
        auto collisionPairs = globalGrid.getPotentialInterObjectCollisionPairs(maxRadius * 2.0);
        
        // 16. Prepare to sort collisions by distance for priority
        std::vector<std::tuple<size_t, size_t, double>> sortedPairs;
        sortedPairs.reserve(collisionPairs.size());
        
        // 17. Calculate distances and filter out non-colliding pairs
        for (const auto& [id1, id2] : collisionPairs) {
            if (id1 >= particleMap.size() || id2 >= particleMap.size()) continue;
            
            const ParticleInfo& info1 = particleMap[id1];
            const ParticleInfo& info2 = particleMap[id2];
            
            auto& p1 = objects[info1.objectIndex]->getParticles()[info1.localIndex];
            auto& p2 = objects[info2.objectIndex]->getParticles()[info2.localIndex];
            
            double distSq = (p2.position - p1.position).squaredNorm();
            double minDist = info1.radius + info2.radius;
            double minDistSq = minDist * minDist;
            
            // 18. Only keep pairs that are actually touching
            if (distSq < minDistSq * params.collisionCullDistance && distSq > 1e-12) {
                sortedPairs.push_back({id1, id2, distSq});
            }
        }
        
        // 19. Sort closest collisions first for most important responses
        std::sort(sortedPairs.begin(), sortedPairs.end(), 
                  [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
        
        // 20. Track how many collisions each particle has processed
        std::vector<int> particleCollisionCount(particleMap.size(), 0);
        
        // 21. Process each collision in order of importance
        for (const auto& [id1, id2, distSq] : sortedPairs) {
            // 22. Skip particles that have reached their collision limit
            if (id1 >= particleCollisionCount.size() || id2 >= particleCollisionCount.size() ||
                particleCollisionCount[id1] >= params.maxCollisionPairsPerParticle ||
                particleCollisionCount[id2] >= params.maxCollisionPairsPerParticle) {
                continue;
            }
            
            const ParticleInfo& info1 = particleMap[id1];
            const ParticleInfo& info2 = particleMap[id2];
            
            // 21. Get actual particle objects
            auto& particles1 = objects[info1.objectIndex]->getParticles();
            auto& particles2 = objects[info2.objectIndex]->getParticles();
            
            if (info1.localIndex >= particles1.size() || info2.localIndex >= particles2.size()) continue;
            
            Particle& p1 = particles1[info1.localIndex];
            Particle& p2 = particles2[info2.localIndex];
            
            // 22. Calculate collision details
            Eigen::Vector3d diff = p2.position - p1.position;
            double minDist = (info1.radius + info2.radius) * params.contactOffset;
            
            // 23. Calculate collision direction and overlap
            double dist = std::sqrt(distSq);
            double invDist = 1.0 / dist;
            Eigen::Vector3d normal = diff * invDist;
            double penetration = minDist - dist;
            
            // 24. Calculate mass ratios for realistic response
            double totalMass = p1.mass + p2.mass;
            double invTotalMass = 1.0 / totalMass;
            double mass1Ratio = p2.mass * invTotalMass;
            double mass2Ratio = p1.mass * invTotalMass;
            
            // 25. Separate overlapping particles
            double correctionMagnitude = std::min(
                penetration * params.separationStrength * relaxationFactor,
                static_cast<double>(params.maxPositionCorrection)
            );
            
            Eigen::Vector3d correction = normal * correctionMagnitude;
            p1.position -= correction * mass1Ratio;
            p2.position += correction * mass2Ratio;
            
            // 26. Adjust velocities if particles are moving toward each other
            double relVel = (p2.velocity - p1.velocity).dot(normal);
            if (relVel < 0) {
                double impulse = -relVel * (1.0 + params.restitution) * relaxationFactor;
                Eigen::Vector3d velocityChange = normal * impulse;
                p1.velocity -= velocityChange * mass1Ratio;
                p2.velocity += velocityChange * mass2Ratio;
            }
            
            // 27. Track that these particles have been processed
            particleCollisionCount[id1]++;
            particleCollisionCount[id2]++;
        }
    }
    
    // 28. Update extended position data for all objects
    for (auto& object : objects) {
        if (object) {
            Particle::batchUpdateExtendedPositions(object->getParticles());
        }
    }
}

/**
 * @brief Hierarchical collision detection using adaptive octree structure
 * @param objects Vector of deformable objects for collision detection
 * @note Best for varying particle densities and dense collision scenarios
 */
void PhysicsEngine::handleInterObjectCollisionsWithOctree(
    std::vector<std::unique_ptr<DeformableObject>>& objects) {
    
    // 1. Count total particles to adjust performance
    size_t totalParticles = 0;
    for (const auto& obj : objects) {
        if (obj) totalParticles += obj->getParticles().size();
    }
    
    // 2. Use fewer iterations for large simulations
    int numIterations = 4;  
    if (params.useAdaptiveIterations) {
        if (totalParticles > 10000) numIterations = 1;
        else if (totalParticles > 5000) numIterations = 2;
        else if (totalParticles > 2000) numIterations = 3;
        else if (totalParticles > 1000) numIterations = 3;  
        else if (totalParticles > 500) numIterations = 4;   
    }
    
    const double relaxationFactor = 1.0;
    
    // 3. Skip if no collisions possible
    if (objects.size() <= 1) return;
    
    // 4. Use simple method for small simulations
    if (totalParticles < 100) {
        handleInterObjectCollisions(objects);
        return;
    }
    
    // 5. Set up tracking for particle information
    struct ParticleInfo {
        size_t objectIndex;
        size_t localIndex;
        double radius;
    };
    std::vector<ParticleInfo> particleMap;
    
    // 6. Build mapping from global particle IDs to objects
    size_t globalParticleId = 0;
    for (size_t objIdx = 0; objIdx < objects.size(); ++objIdx) {
        if (!objects[objIdx]) continue;
        
        // 7. Update object boundaries if needed
        if (objects[objIdx]->needsBoundingSphereUpdate()) {
            objects[objIdx]->updateBoundingSphere();
        }
        
        auto& particles = objects[objIdx]->getParticles();
        double radius = objects[objIdx]->getAdaptiveParticleRadius();
        
        // 8. Store information for each particle
        for (size_t localIdx = 0; localIdx < particles.size(); ++localIdx) {
            particleMap.push_back({objIdx, localIdx, radius});
            globalParticleId++;
        }
    }
    
    // 9. Create octree for hierarchical collision detection
    OctreeCollisionDetection octree;
    
    // 10. Perform multiple collision resolution passes
    for (int iter = 0; iter < numIterations; ++iter) {
        // 11. Build tree structure with current particle positions
        octree.build(objects);
        
        // 12. Find the largest particle radius
        double maxRadius = 0.0;
        for (const auto& obj : objects) {
            if (obj) {
                maxRadius = std::max(maxRadius, obj->getAdaptiveParticleRadius());
            }
        }
        
        // 13. Get potential collision pairs from tree
        auto collisionPairs = octree.getPotentialCollisionPairs(maxRadius * 2.0);
        
        // 14. Prepare to sort collisions by distance for priority
        std::vector<std::tuple<size_t, size_t, double>> sortedPairs;
        sortedPairs.reserve(collisionPairs.size());
        
        // 15. Calculate distances and filter out non-colliding pairs
        for (const auto& [id1, id2] : collisionPairs) {
            if (id1 >= particleMap.size() || id2 >= particleMap.size()) continue;
            
            const ParticleInfo& info1 = particleMap[id1];
            const ParticleInfo& info2 = particleMap[id2];
            
            auto& p1 = objects[info1.objectIndex]->getParticles()[info1.localIndex];
            auto& p2 = objects[info2.objectIndex]->getParticles()[info2.localIndex];
            
            double distSq = (p2.position - p1.position).squaredNorm();
            double minDist = info1.radius + info2.radius;
            double minDistSq = minDist * minDist;
            
            // 16. Only keep pairs that are actually touching
            if (distSq < minDistSq * params.collisionCullDistance && distSq > 1e-12) {
                sortedPairs.push_back({id1, id2, distSq});
            }
        }
        
        // 17. Sort closest collisions first for most important responses
        std::sort(sortedPairs.begin(), sortedPairs.end(), 
                  [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
        
        // 18. Track how many collisions each particle has processed
        std::vector<int> particleCollisionCount(particleMap.size(), 0);
        
        // 19. Process each collision in order of importance
        for (const auto& [id1, id2, distSq] : sortedPairs) {
            // 20. Skip particles that have reached their collision limit
            if (id1 >= particleCollisionCount.size() || id2 >= particleCollisionCount.size() ||
                particleCollisionCount[id1] >= params.maxCollisionPairsPerParticle ||
                particleCollisionCount[id2] >= params.maxCollisionPairsPerParticle) {
                continue;
            }
            
            const ParticleInfo& info1 = particleMap[id1];
            const ParticleInfo& info2 = particleMap[id2];
            
            // 21. Get actual particle objects
            auto& particles1 = objects[info1.objectIndex]->getParticles();
            auto& particles2 = objects[info2.objectIndex]->getParticles();
            
            if (info1.localIndex >= particles1.size() || info2.localIndex >= particles2.size()) continue;
            
            Particle& p1 = particles1[info1.localIndex];
            Particle& p2 = particles2[info2.localIndex];
            
            // 22. Calculate collision details
            Eigen::Vector3d diff = p2.position - p1.position;
            double minDist = (info1.radius + info2.radius) * params.contactOffset;
            
            // 23. Calculate collision direction and overlap
            double dist = std::sqrt(distSq);
            double invDist = 1.0 / dist;
            Eigen::Vector3d normal = diff * invDist;
            double penetration = minDist - dist;
            
            // 24. Calculate mass ratios for realistic response
            double totalMass = p1.mass + p2.mass;
            double invTotalMass = 1.0 / totalMass;
            double mass1Ratio = p2.mass * invTotalMass;
            double mass2Ratio = p1.mass * invTotalMass;
            
            // 25. Separate overlapping particles
            double correctionMagnitude = std::min(
                penetration * params.separationStrength * relaxationFactor,
                static_cast<double>(params.maxPositionCorrection)
            );
            
            Eigen::Vector3d correction = normal * correctionMagnitude;
            p1.position -= correction * mass1Ratio;
            p2.position += correction * mass2Ratio;
            
            // 26. Adjust velocities if particles are moving toward each other
            double relVel = (p2.velocity - p1.velocity).dot(normal);
            if (relVel < 0) {
                double impulse = -relVel * (1.0 + params.restitution) * relaxationFactor;
                Eigen::Vector3d velocityChange = normal * impulse;
                p1.velocity -= velocityChange * mass1Ratio;
                p2.velocity += velocityChange * mass2Ratio;
            }
            
            // 27. Track that these particles have been processed
            particleCollisionCount[id1]++;
            particleCollisionCount[id2]++;
        }
    }
    
    // 28. Update extended position data for all objects
    for (auto& object : objects) {
        if (object) {
            Particle::batchUpdateExtendedPositions(object->getParticles());
        }
    }
}