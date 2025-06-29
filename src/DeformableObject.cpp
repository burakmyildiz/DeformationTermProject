//DeformableObject.cpp

#include "DeformableObject.h"
#include <polyscope/polyscope.h>
#include <polyscope/point_cloud.h>
#include <polyscope/surface_mesh.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>

#ifdef USE_OPENMP
#include <omp.h>
#endif

/**
 * @brief Constructs deformable object with unique identifier
 * @param name Unique name for Polyscope visualization (default: "deformable")
 */
DeformableObject::DeformableObject(const std::string& name) 
    : pointCloudName(name + "_points"), meshName(name + "_mesh"), 
      boundingSphereNeedsUpdate(true), adaptiveParticleRadius(0.25) {
}

/**
 * @brief Resets object to initial configuration
 * Restores all particles to rest positions and clears velocities
 */
void DeformableObject::reset() {
    for (auto& particle : particles) {
        particle.reset();
    }
    boundingSphereNeedsUpdate = true;
}

/**
 * @brief Performs one physics simulation step
 * @param engine Physics engine containing simulation parameters and methods
 * @param dt Time step duration for integration
 * @note Applies shape matching, integrates motion, handles collisions
 */
void DeformableObject::update(PhysicsEngine& engine, double dt) {
    if (engine.getParameters().isPaused) {
        return;
    }
    
    // 1. Apply external forces like gravity to particle velocities
    engine.applyGravity(particles);
    
    // 2. Apply velocity damping early for stability
    engine.applyDamping(particles);
    
    // 3. Compute optimal shape transformation using weighted deformation modes
    ShapeMatcher::MatchResult matchResult = shapeMatcher.computeRigidTransformation(
        particles, 
        engine.getParameters().alpha,
        engine.getParameters().beta,
        engine.getParameters().gamma);
    
    // 4. Push particles toward their goal positions by adjusting velocities
    engine.applyShapeMatching(particles, matchResult.goalPositions, dt);
    
    // 5. Integrate particle positions forward in time
    engine.updatePositions(particles, dt);
    
    // 6. REMOVED: adjustForGroundContact - this was preventing collisions!
    
    // 7. Resolve any collisions with ground, walls, or other objects
    engine.handleCollisions(particles);
    
    // 8. Mark bounding volume as needing recalculation
    boundingSphereNeedsUpdate = true;
    
    // 9. Refresh visual representation for current frame
    updateVisualization();
}

/**
 * @brief Initializes Polyscope visualization components
 * Creates point cloud and mesh representations for rendering
 */
void DeformableObject::initializeVisualization() {
    // 1. Extract current particle positions
    std::vector<Eigen::Vector3d> positions;
    for (const auto& p : particles) {
        positions.push_back(p.position);
    }
    
    // 2. Choose appropriate visualization based on available geometry
    if (!triangles.empty()) {
        std::vector<std::array<size_t, 3>> faces;
        for (const auto& tri : triangles) {
            faces.push_back({static_cast<size_t>(tri[0]), 
                           static_cast<size_t>(tri[1]), 
                           static_cast<size_t>(tri[2])});
        }
        
        polyscope::registerSurfaceMesh(meshName, positions, faces);
        
        polyscope::registerPointCloud(pointCloudName, positions);
        polyscope::getPointCloud(pointCloudName)->setEnabled(false);
    } else {
        polyscope::registerPointCloud(pointCloudName, positions);
    }
}

/**
 * @brief Updates Polyscope visualization with current particle positions
 * Refreshes point cloud and mesh positions for current frame
 */
void DeformableObject::updateVisualization() {
    // 1. Collect current particle positions
    std::vector<Eigen::Vector3d> positions;
    for (const auto& p : particles) {
        positions.push_back(p.position);
    }
    
    // 2. Update registered visualizations with new positions
    if (polyscope::hasPointCloud(pointCloudName)) {
        polyscope::getPointCloud(pointCloudName)->updatePointPositions(positions);
    }
    
    if (!triangles.empty() && polyscope::hasSurfaceMesh(meshName)) {
        polyscope::getSurfaceMesh(meshName)->updateVertexPositions(positions);
    }
}

/**
 * @brief Loads object geometry from OBJ file with scaling and positioning
 * @param filename Path to OBJ file containing mesh data
 * @param c 3D position to center the object (default: origin)
 * @param s Uniform scaling factor for the mesh (default: 1.0)
 * @return True if loading succeeded, false on file error or parsing failure
 */
bool DeformableObject::loadFromOBJ(const std::string& filename, const Eigen::Vector3d& c, double s) {
    center = c;
    scale = s;
    particles.clear();
    triangles.clear();
    
    // 1. Open and validate file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << filename << std::endl;
        return false;
    }
    
    // 2. Parse OBJ file line by line
    std::vector<Eigen::Vector3d> vertices;
    std::string line;
    
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        
        if (prefix == "v") {
            double x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(Eigen::Vector3d(x, y, z));
        }
        else if (prefix == "f") {
            std::vector<int> faceIndices;
            std::string vertex;
            
            while (iss >> vertex) {
                int idx = std::stoi(vertex.substr(0, vertex.find('/'))) - 1;
                faceIndices.push_back(idx);
            }
            
            if (faceIndices.size() >= 3) {
                for (size_t i = 1; i < faceIndices.size() - 1; ++i) {
                    triangles.push_back({faceIndices[0], faceIndices[i], faceIndices[i + 1]});
                }
            }
        }
    }
    
    file.close();
    
    // 3. Create particles from vertices with scaling and positioning
    for (const auto& v : vertices) {
        Eigen::Vector3d pos = center + v * scale;
        particles.emplace_back(pos, 1.0);
    }
    
    // 4. Calculate adaptive collision radius based on mesh complexity
    double vertexToFaceRatio = vertices.empty() ? 1.0 : 
        static_cast<double>(vertices.size()) / std::max(1.0, static_cast<double>(triangles.size()));
    
    double baseRadius = 0.25;
    if (vertexToFaceRatio > 0.8) {
        adaptiveParticleRadius = baseRadius * 1.5;  
    } else if (vertexToFaceRatio > 0.6) {
        adaptiveParticleRadius = baseRadius * 1.2;  
    } else {
        adaptiveParticleRadius = baseRadius;        
    }
    
    std::cout << "Loaded OBJ with " << particles.size() << " vertices and " 
              << triangles.size() << " triangles" << std::endl;
    std::cout << "Vertex-to-face ratio: " << vertexToFaceRatio 
              << ", Adaptive radius: " << adaptiveParticleRadius << std::endl;
    
    return true;
}

/**
 * @brief Applies external force to particles within radius of origin point
 * @param origin 3D center point for force application
 * @param force 3D force vector to apply
 * @param radius Distance within which particles are affected
 * @note Force falls off with distance from origin point
 */
void DeformableObject::applyForce(const Eigen::Vector3d& origin, const Eigen::Vector3d& force, double radius) {
    #ifdef USE_OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < particles.size(); ++i) {
        auto& particle = particles[i];
        // 1. Calculate distance from force origin
        Eigen::Vector3d diff = particle.position - origin;
        double dist = diff.norm();
        
        // 2. Apply force with distance-based falloff
        if (dist < radius) {
            double falloff = 1.0 - (dist / radius);
            particle.velocity += force * falloff;
        }
    }
}

/**
 * @brief Updates bounding sphere for broad-phase collision culling
 * Recalculates center and radius from current particle positions
 */
void DeformableObject::updateBoundingSphere() {
    if (particles.empty()) {
        boundingCenter = Eigen::Vector3d::Zero();
        boundingRadius = 0.0;
        return;
    }
    
    // 1. Calculate geometric center of all particles
    boundingCenter = Eigen::Vector3d::Zero();
    for (const auto& particle : particles) {
        boundingCenter += particle.position;
    }
    boundingCenter /= static_cast<double>(particles.size());
    
    // 2. Find the particle farthest from center
    boundingRadius = 0.0;
    #ifdef USE_OPENMP
    #pragma omp parallel for reduction(max:boundingRadius)
    #endif
    for (size_t i = 0; i < particles.size(); ++i) {
        double dist = (particles[i].position - boundingCenter).norm();
        if (dist > boundingRadius) {
            boundingRadius = dist;
        }
    }
    
    // 3. Add safety margin for collision detection
    boundingRadius += 0.15;
    
    boundingSphereNeedsUpdate = false;
}

/**
 * @brief Adjusts entire object position to prevent ground penetration
 * @param groundHeight Y-coordinate of the ground plane
 * @param particleRadius Radius of particles for collision detection
 * @note Lifts entire object uniformly to maintain shape integrity
 */
void DeformableObject::adjustForGroundContact(double groundHeight, double particleRadius) {
    // 1. Find the lowest point of the object
    double lowestY = std::numeric_limits<double>::max();
    for (const auto& particle : particles) {
        lowestY = std::min(lowestY, particle.position.y());
    }
    
    // 2. Calculate ground penetration
    double groundLevel = groundHeight + particleRadius;
    if (lowestY < groundLevel) {
        double lift = groundLevel - lowestY;
        
        // 3. Lift all particles to maintain shape
        for (auto& particle : particles) {
            particle.position.y() += lift;
        }
    }
}
