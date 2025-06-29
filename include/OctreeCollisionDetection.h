#pragma once
#include <vector>
#include <memory>
#include <array>
#include <Eigen/Dense>
#include "Particle.h"

class DeformableObject;

/**
 * @brief Hierarchical octree node for spatial collision detection
 * Implements adaptive subdivision based on particle density for efficient collision detection
 */
class OctreeNode {
public:
    struct ParticleEntry {
        size_t globalId;
        size_t objectId;
        Eigen::Vector3d position;
    };
    
private:
    static constexpr int MAX_PARTICLES_PER_LEAF = 8;
    static constexpr int MAX_DEPTH = 6;
    static constexpr float MIN_NODE_SIZE = 0.1f;
    
    Eigen::Vector3d center;
    float halfSize;
    int depth;
    
    std::vector<ParticleEntry> particles;
    std::array<std::unique_ptr<OctreeNode>, 8> children;
    bool isLeaf;
    
public:
    /**
     * @brief Constructs octree node with bounding box and depth
     * @param center 3D center point of the node's bounding box
     * @param halfSize Half-width of the node's bounding box
     * @param depth Current depth in the octree (0 = root)
     */
    OctreeNode(const Eigen::Vector3d& center, float halfSize, int depth = 0)
        : center(center), halfSize(halfSize), depth(depth), isLeaf(true) {}
    
    /**
     * @brief Inserts particle into octree, subdividing if necessary
     * @param particle Particle data including ID, object ID, and position
     * @note Automatically subdivides when exceeding MAX_PARTICLES_PER_LEAF
     */
    void insert(const ParticleEntry& particle) {
        if (isLeaf && (particles.size() < MAX_PARTICLES_PER_LEAF || 
                       depth >= MAX_DEPTH || halfSize <= MIN_NODE_SIZE)) {
            particles.push_back(particle);
            return;
        }
        
        if (isLeaf) {
            subdivide();
        }
        
        int childIndex = getChildIndex(particle.position);
        children[childIndex]->insert(particle);
    }
    
    /**
     * @brief Finds all particles within radius of a query point
     * @param queryPos 3D center position for radius search
     * @param radius Search radius from query position
     * @param results Output vector to store particle IDs within radius
     * @note Uses sphere-box intersection for early culling optimization
     */
    void queryRadius(const Eigen::Vector3d& queryPos, float radius,
                    std::vector<size_t>& results) const {
        float sqDist = 0.0f;
        for (int i = 0; i < 3; ++i) {
            float d = std::abs(queryPos[i] - center[i]) - halfSize;
            if (d > 0) sqDist += d * d;
        }
        if (sqDist > radius * radius) return;
        
        if (isLeaf) {
            float radiusSq = radius * radius;
            for (const auto& p : particles) {
                if ((p.position - queryPos).squaredNorm() <= radiusSq) {
                    results.push_back(p.globalId);
                }
            }
        } else {
            for (const auto& child : children) {
                if (child) child->queryRadius(queryPos, radius, results);
            }
        }
    }
    
    /**
     * @brief Gets potential collision pairs between different objects within radius
     * @param collisionRadius Maximum distance for potential collisions
     * @param pairs Output vector to store particle ID pairs that might collide
     * @note Filters out same-object pairs automatically for performance
     */
    void getPotentialCollisionPairs(float collisionRadius,
                                  std::vector<std::pair<size_t, size_t>>& pairs) const {
        if (isLeaf) {
            float radiusSq = collisionRadius * collisionRadius;
            for (size_t i = 0; i < particles.size(); ++i) {
                for (size_t j = i + 1; j < particles.size(); ++j) {
                    // Skip if same object
                    if (particles[i].objectId == particles[j].objectId) continue;
                    
                    if ((particles[i].position - particles[j].position).squaredNorm() <= radiusSq) {
                        pairs.push_back({particles[i].globalId, particles[j].globalId});
                    }
                }
            }
        } else {
            for (const auto& child : children) {
                if (child) child->getPotentialCollisionPairs(collisionRadius, pairs);
            }
            

            checkBetweenChildren(collisionRadius, pairs);
        }
    }
    
private:
    /**
     * @brief Subdivides leaf node into 8 children and redistributes particles
     * @note Called automatically when particle count exceeds MAX_PARTICLES_PER_LEAF
     */
    void subdivide() {
        isLeaf = false;
        float childSize = halfSize * 0.5f;
        
        for (int i = 0; i < 8; ++i) {
            Eigen::Vector3d childCenter = center;
            childCenter.x() += ((i & 1) ? childSize : -childSize);
            childCenter.y() += ((i & 2) ? childSize : -childSize);
            childCenter.z() += ((i & 4) ? childSize : -childSize);
            
            children[i] = std::make_unique<OctreeNode>(childCenter, childSize, depth + 1);
        }
        
        for (const auto& p : particles) {
            int childIndex = getChildIndex(p.position);
            children[childIndex]->insert(p);
        }
        particles.clear();
    }
    
    /**
     * @brief Determines which child octant contains the given position
     * @param pos 3D position to locate within children
     * @return Integer index [0-7] of the child octant containing position
     */
    int getChildIndex(const Eigen::Vector3d& pos) const {
        int index = 0;
        if (pos.x() > center.x()) index |= 1;
        if (pos.y() > center.y()) index |= 2;
        if (pos.z() > center.z()) index |= 4;
        return index;
    }
    
    /**
     * @brief Checks for collision pairs between particles in different child nodes
     * @param collisionRadius Maximum distance for potential collisions
     * @param pairs Output vector to accumulate collision pairs
     * @note Handles cross-boundary collisions between adjacent octree children
     */
    void checkBetweenChildren(float collisionRadius, 
                             std::vector<std::pair<size_t, size_t>>& pairs) const {
        float radiusSq = collisionRadius * collisionRadius;
        

        static const int adjacentPairs[][2] = {
            // Bottom level face adjs
            {0, 1}, {1, 3}, {3, 2}, {2, 0},
            // Top level face adjs  
            {4, 5}, {5, 7}, {7, 6}, {6, 4},
            // Vertical adjs between levels
            {0, 4}, {1, 5}, {2, 6}, {3, 7},
            // Diagonal edge adjs
            {0, 5}, {1, 4}, {2, 7}, {3, 6},
            {1, 2}, {0, 3}, {4, 7}, {5, 6}
        };
        
        constexpr int numAdjacencies = sizeof(adjacentPairs) / sizeof(adjacentPairs[0]);
        
        for (int k = 0; k < numAdjacencies; ++k) {
            int i = adjacentPairs[k][0];
            int j = adjacentPairs[k][1];
            
            if (!children[i] || !children[j]) continue;
            
            checkParticlesBetweenNodes(*children[i], *children[j], radiusSq, pairs);
        }
    }
    
    /**
     * @brief Performs brute-force collision check between particles in two nodes
     * @param node1 First octree node containing particles
     * @param node2 Second octree node containing particles
     * @param radiusSq Squared collision radius for distance comparison
     * @param pairs Output vector to accumulate collision pairs
     * @note Called for adjacent nodes to handle cross-boundary collisions
     */
    void checkParticlesBetweenNodes(const OctreeNode& node1, const OctreeNode& node2,
                                   float radiusSq, std::vector<std::pair<size_t, size_t>>& pairs) const {
        std::vector<ParticleEntry> particles1, particles2;
        node1.collectParticles(particles1);
        node2.collectParticles(particles2);
        
        for (const auto& p1 : particles1) {
            for (const auto& p2 : particles2) {
                if (p1.objectId == p2.objectId) continue;
                
                if ((p1.position - p2.position).squaredNorm() <= radiusSq) {
                    pairs.push_back({p1.globalId, p2.globalId});
                }
            }
        }
    }
    
    /**
     * @brief Recursively collects all particles from this node and its children
     * @param allParticles Output vector to accumulate particle entries
     * @note Used for gathering particles from subtrees for collision checking
     */
    void collectParticles(std::vector<ParticleEntry>& allParticles) const {
        if (isLeaf) {
            allParticles.insert(allParticles.end(), particles.begin(), particles.end());
        } else {
            for (const auto& child : children) {
                if (child) child->collectParticles(allParticles);
            }
        }
    }
};

/**
 * @brief High-level wrapper for octree-based collision detection system
 * Provides simplified interface for building octrees and querying collisions
 */
class OctreeCollisionDetection {
private:
    std::unique_ptr<OctreeNode> root;
    Eigen::Vector3d sceneMin, sceneMax;
    
public:
    /**
     * @brief Builds octree from all particles in the given deformable objects
     * @param objects Vector of deformable objects to include in octree
     * @note Automatically calculates scene bounds and creates root node
     */
    void build(const std::vector<std::unique_ptr<DeformableObject>>& objects) {
        sceneMin = Eigen::Vector3d(1e10, 1e10, 1e10);
        sceneMax = Eigen::Vector3d(-1e10, -1e10, -1e10);
        
        for (const auto& obj : objects) {
            if (!obj) continue;
            const auto& particles = obj->getParticles();
            for (const auto& p : particles) {
                sceneMin = sceneMin.cwiseMin(p.position);
                sceneMax = sceneMax.cwiseMax(p.position);
            }
        }
        
        Eigen::Vector3d center = (sceneMin + sceneMax) * 0.5;
        float halfSize = (sceneMax - sceneMin).maxCoeff() * 0.6f;
        
        root = std::make_unique<OctreeNode>(center, halfSize);
        
        size_t globalId = 0;
        for (size_t objId = 0; objId < objects.size(); ++objId) {
            if (!objects[objId]) continue;
            const auto& particles = objects[objId]->getParticles();
            
            for (const auto& p : particles) {
                OctreeNode::ParticleEntry entry{globalId++, objId, p.position};
                root->insert(entry);
            }
        }
    }
    
    /**
     * @brief Gets all potential collision pairs within collision radius
     * @param collisionRadius Maximum distance for potential collisions
     * @return Vector of particle ID pairs that might be colliding
     * @note Excludes same-object pairs for performance optimization
     */
    std::vector<std::pair<size_t, size_t>> getPotentialCollisionPairs(float collisionRadius) {
        std::vector<std::pair<size_t, size_t>> pairs;
        if (root) {
            root->getPotentialCollisionPairs(collisionRadius, pairs);
        }
        return pairs;
    }
    
    /**
     * @brief Finds all particles within radius of a query position
     * @param position 3D center position for radius search
     * @param radius Search radius from query position
     * @return Vector of particle IDs within the specified radius
     * @note Uses hierarchical pruning for efficient spatial queries
     */
    std::vector<size_t> queryRadius(const Eigen::Vector3d& position, float radius) {
        std::vector<size_t> results;
        if (root) {
            root->queryRadius(position, radius, results);
        }
        return results;
    }
};