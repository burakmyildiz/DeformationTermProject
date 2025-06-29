#include "SpatialHashGrid.h"
#include <cmath>
#include <algorithm>
#include <set>

/**
 * @brief Constructs spatial hash grid with specified cell size
 * @param cellSize Size of each grid cell (affects performance vs accuracy)
 */
SpatialHashGrid::SpatialHashGrid(float cellSize) 
    : cellSize_(cellSize), invCellSize_(1.0f / cellSize) {
}

/**
 * @brief Clears all particles from the grid
 * Resets grid state for new frame or object set
 */
void SpatialHashGrid::clear() {
    grid_.clear();
    particlePositions_.clear();
    particleData_.clear();
}

/**
 * @brief Converts 3D position to grid cell coordinates
 * @param position 3D world position
 * @return GridCell containing integer cell coordinates
 */
SpatialHashGrid::GridCell SpatialHashGrid::getGridCell(const Eigen::Vector3d& position) const {
    return {
        static_cast<int>(std::floor(position.x() * invCellSize_)),
        static_cast<int>(std::floor(position.y() * invCellSize_)),
        static_cast<int>(std::floor(position.z() * invCellSize_))
    };
}

/**
 * @brief Inserts particle into grid at specified position
 * @param particleId Unique identifier for the particle
 * @param position 3D world position of the particle
 */
void SpatialHashGrid::insertParticle(size_t particleId, const Eigen::Vector3d& position) {
    // 1. Find which grid cell this particle belongs in
    GridCell cell = getGridCell(position);
    grid_[cell].insert(particleId);
    
    // 2. Keep track of particle position for distance calculations
    if (particleId >= particlePositions_.size()) {
        particlePositions_.resize(particleId + 1);
    }
    particlePositions_[particleId] = position;
}

/**
 * @brief Inserts particle with object ID for inter-object collision filtering
 * @param particleId Unique identifier for the particle
 * @param objectId ID of the object this particle belongs to
 * @param position 3D world position of the particle
 */
void SpatialHashGrid::insertParticleWithObjectId(size_t particleId, size_t objectId, const Eigen::Vector3d& position) {
    // 1. Find which grid cell this particle belongs in
    GridCell cell = getGridCell(position);
    grid_[cell].insert(particleId);
    
    // 2. Keep track of particle position for distance calculations
    if (particleId >= particlePositions_.size()) {
        particlePositions_.resize(particleId + 1);
    }
    particlePositions_[particleId] = position;
    
    // 3. Store which object this particle belongs to
    if (particleId >= particleData_.size()) {
        particleData_.resize(particleId + 1);
    }
    particleData_[particleId] = {particleId, objectId, position};
}

/**
 * @brief Builds grid from particle vector (convenience method)
 * @param particles Vector of particles to insert into grid
 */
void SpatialHashGrid::build(const std::vector<Particle>& particles) {
    // 1. Start fresh and prepare space for all particles
    clear();
    particlePositions_.reserve(particles.size());
    
    // 2. Add each particle to the grid
    for (size_t i = 0; i < particles.size(); ++i) {
        insertParticle(i, particles[i].position);
    }
}

/**
 * @brief Gets neighboring cells within specified radius
 * @param cell Center cell for neighborhood query
 * @param radius Number of cells to include in each direction
 * @return Vector of GridCell coordinates in the neighborhood
 */
std::vector<SpatialHashGrid::GridCell> SpatialHashGrid::getNeighborCells(
    const GridCell& cell, int radius) const {
    
    // 1. Prepare space for all neighboring cells
    std::vector<GridCell> neighbors;
    neighbors.reserve((2 * radius + 1) * (2 * radius + 1) * (2 * radius + 1));
    
    // 2. Generate all cells within the specified radius
    for (int dx = -radius; dx <= radius; ++dx) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dz = -radius; dz <= radius; ++dz) {
                neighbors.push_back({
                    cell.x + dx,
                    cell.y + dy,
                    cell.z + dz
                });
            }
        }
    }
    
    return neighbors;
}

/**
 * @brief Finds all particles within radius of query position
 * @param position Center position for radius query
 * @param radius Search radius
 * @return Vector of particle IDs within the specified radius
 */
std::vector<size_t> SpatialHashGrid::queryRadius(
    const Eigen::Vector3d& position, float radius) const {
    
    std::vector<size_t> result;
    
    // 1. Figure out how many grid cells the search radius covers
    int cellRadius = static_cast<int>(std::ceil(radius * invCellSize_));
    
    // 2. Find the center cell and get all nearby cells
    GridCell centerCell = getGridCell(position);
    std::vector<GridCell> neighborCells = getNeighborCells(centerCell, cellRadius);
    
    std::unordered_set<size_t> uniqueParticles;
    
    // 3. Check each nearby cell for particles within range
    for (const auto& cell : neighborCells) {
        auto it = grid_.find(cell);
        if (it != grid_.end()) {
            for (size_t particleId : it->second) {
                if (particleId < particlePositions_.size()) {
                    double distance = (particlePositions_[particleId] - position).norm();
                    if (distance <= radius) {
                        uniqueParticles.insert(particleId);
                    }
                }
            }
        }
    }
    
    result.assign(uniqueParticles.begin(), uniqueParticles.end());
    return result;
}

/**
 * @brief Gets all potential collision pairs including same-object particles (legacy)
 * @param collisionRadius Maximum distance for potential collisions
 * @return Vector of particle ID pairs that might be colliding
 */
std::vector<std::pair<size_t, size_t>> SpatialHashGrid::getPotentialCollisionPairs(
    float collisionRadius) const {
    
    std::vector<std::pair<size_t, size_t>> pairs;
    std::set<std::pair<size_t, size_t>> uniquePairs;
    
    // 1. Go through each grid cell to find potential collisions
    for (const auto& [cell, particleSet] : grid_) {
        std::vector<size_t> particles_in_cell(particleSet.begin(), particleSet.end());
        
        // 2. Check particles within the same cell against each other
        for (size_t i = 0; i < particles_in_cell.size(); ++i) {
            for (size_t j = i + 1; j < particles_in_cell.size(); ++j) {
                size_t id1 = particles_in_cell[i];
                size_t id2 = particles_in_cell[j];
                
                if (id1 < particlePositions_.size() && id2 < particlePositions_.size()) {
                    double distanceSq = (particlePositions_[id1] - particlePositions_[id2]).squaredNorm();
                    if (distanceSq <= (collisionRadius * collisionRadius)) {
                        uniquePairs.insert({std::min(id1, id2), std::max(id1, id2)});
                    }
                }
            }
        }
        
        // 3. Check particles against neighboring cells without duplicates
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx < 0 || (dx == 0 && dy < 0) || (dx == 0 && dy == 0 && dz <= 0)) {
                        continue;
                    }

                    GridCell neighborCoords = {cell.x + dx, cell.y + dy, cell.z + dz};
                    auto neighborIt = grid_.find(neighborCoords);
                    
                    if (neighborIt != grid_.end()) {
                        const auto& neighborParticleSet = neighborIt->second;
                        for (size_t id1 : particleSet) {
                            for (size_t id2 : neighborParticleSet) {
                                if (id1 < particlePositions_.size() && id2 < particlePositions_.size()) {
                                    double distanceSq = (particlePositions_[id1] - particlePositions_[id2]).squaredNorm();
                                    if (distanceSq <= (collisionRadius * collisionRadius)) {
                                        uniquePairs.insert({std::min(id1, id2), std::max(id1, id2)});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    pairs.assign(uniquePairs.begin(), uniquePairs.end());
    return pairs;
}

/**
 * @brief Gets collision pairs between different objects only (performance optimized)
 * @param collisionRadius Maximum distance for potential collisions
 * @return Vector of inter-object particle pairs that might be colliding
 * @note Filters out intra-object collisions early for better performance
 */
std::vector<std::pair<size_t, size_t>> SpatialHashGrid::getPotentialInterObjectCollisionPairs(
    float collisionRadius) const {
    
    std::vector<std::pair<size_t, size_t>> pairs;
    std::set<std::pair<size_t, size_t>> uniquePairs;
    
    // 1. Go through each grid cell to find inter-object collisions only
    for (const auto& [cell, particleSet] : grid_) {
        std::vector<size_t> particles_in_cell(particleSet.begin(), particleSet.end());
        
        // 2. Check particles within the same cell, skipping same-object pairs
        for (size_t i = 0; i < particles_in_cell.size(); ++i) {
            for (size_t j = i + 1; j < particles_in_cell.size(); ++j) {
                size_t id1 = particles_in_cell[i];
                size_t id2 = particles_in_cell[j];
                
                if (id1 < particleData_.size() && id2 < particleData_.size() &&
                    particleData_[id1].objectId == particleData_[id2].objectId) {
                    continue;
                }
                
                if (id1 < particlePositions_.size() && id2 < particlePositions_.size()) {
                    double distanceSq = (particlePositions_[id1] - particlePositions_[id2]).squaredNorm();
                    if (distanceSq <= (collisionRadius * collisionRadius)) {
                        uniquePairs.insert({std::min(id1, id2), std::max(id1, id2)});
                    }
                }
            }
        }
        
        // 3. Check against neighboring cells, skipping same-object pairs
        for (int dx = -1; dx <= 1; ++dx) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dz = -1; dz <= 1; ++dz) {
                    if (dx < 0 || (dx == 0 && dy < 0) || (dx == 0 && dy == 0 && dz <= 0)) {
                        continue;
                    }

                    GridCell neighborCoords = {cell.x + dx, cell.y + dy, cell.z + dz};
                    auto neighborIt = grid_.find(neighborCoords);
                    
                    if (neighborIt != grid_.end()) {
                        const auto& neighborParticleSet = neighborIt->second;
                        for (size_t id1 : particleSet) {
                            for (size_t id2 : neighborParticleSet) {
                                if (id1 < particleData_.size() && id2 < particleData_.size() &&
                                    particleData_[id1].objectId == particleData_[id2].objectId) {
                                    continue;
                                }
                                
                                if (id1 < particlePositions_.size() && id2 < particlePositions_.size()) {
                                    double distanceSq = (particlePositions_[id1] - particlePositions_[id2]).squaredNorm();
                                    if (distanceSq <= (collisionRadius * collisionRadius)) {
                                        uniquePairs.insert({std::min(id1, id2), std::max(id1, id2)});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    pairs.assign(uniquePairs.begin(), uniquePairs.end());
    return pairs;
}