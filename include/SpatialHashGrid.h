#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <Eigen/Dense>
#include "Particle.h"

class SpatialHashGrid {
public:
    /**
     * @brief Constructs spatial hash grid with specified cell size
     * @param cellSize Size of each grid cell (affects performance vs accuracy)
     */
    SpatialHashGrid(float cellSize);
    
    struct ParticleData {
        size_t globalId;
        size_t objectId;
        Eigen::Vector3d position;
    };
    
    /**
     * @brief Clears all particles from the grid
     * Resets grid state for new frame or object set
     */
    void clear();
    
    /**
     * @brief Inserts particle into grid at specified position
     * @param particleId Unique identifier for the particle
     * @param position 3D world position of the particle
     */
    void insertParticle(size_t particleId, const Eigen::Vector3d& position);
    
    /**
     * @brief Inserts particle with object ID for inter-object collision filtering
     * @param particleId Unique identifier for the particle
     * @param objectId ID of the object this particle belongs to
     * @param position 3D world position of the particle
     */
    void insertParticleWithObjectId(size_t particleId, size_t objectId, const Eigen::Vector3d& position);
    
    /**
     * @brief Builds grid from particle vector (convenience method)
     * @param particles Vector of particles to insert into grid
     */
    void build(const std::vector<Particle>& particles);
    
    /**
     * @brief Finds all particles within radius of query position
     * @param position Center position for radius query
     * @param radius Search radius
     * @return Vector of particle IDs within the specified radius
     */
    std::vector<size_t> queryRadius(const Eigen::Vector3d& position, float radius) const;
    
    /**
     * @brief Gets all potential collision pairs including same-object particles
     * @param collisionRadius Maximum distance for potential collisions
     * @return Vector of particle ID pairs that might be colliding
     */
    std::vector<std::pair<size_t, size_t>> getPotentialCollisionPairs(float collisionRadius) const;
    
    /**
     * @brief Gets collision pairs between different objects only (performance optimized)
     * @param collisionRadius Maximum distance for potential collisions
     * @return Vector of inter-object particle pairs that might be colliding
     * @note Filters out intra-object collisions early for better performance
     */
    std::vector<std::pair<size_t, size_t>> getPotentialInterObjectCollisionPairs(float collisionRadius) const;
    
private:
    struct GridCell {
        int x, y, z;
        
        bool operator==(const GridCell& other) const {
            return x == other.x && y == other.y && z == other.z;
        }
    };
    
    struct GridCellHash {
        size_t operator()(const GridCell& cell) const {
            return std::hash<int>()(cell.x) ^ 
                   (std::hash<int>()(cell.y) << 1) ^ 
                   (std::hash<int>()(cell.z) << 2);
        }
    };
    
    /**
     * @brief Converts 3D position to grid cell coordinates
     * @param position 3D world position
     * @return GridCell containing integer cell coordinates
     */
    GridCell getGridCell(const Eigen::Vector3d& position) const;
    
    /**
     * @brief Gets neighboring cells within specified radius
     * @param cell Center cell for neighborhood query
     * @param radius Number of cells to include in each direction
     * @return Vector of GridCell coordinates in the neighborhood
     */
    std::vector<GridCell> getNeighborCells(const GridCell& cell, int radius) const;
    
    float cellSize_;
    float invCellSize_;
    std::unordered_map<GridCell, std::unordered_set<size_t>, GridCellHash> grid_;
    std::vector<Eigen::Vector3d> particlePositions_;
    std::vector<ParticleData> particleData_;
};