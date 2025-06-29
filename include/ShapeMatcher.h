#pragma once

#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "Particle.h"

class ShapeMatcher {
public:
    struct MatchResult {
        Eigen::Matrix3d rotation;              
        Eigen::Vector3d translation;          
        std::vector<Eigen::Vector3d> goalPositions;
    };
    
    /**
     * @brief Computes optimal shape transformation using blended deformation modes
     * @param particles Vector of particles with current and rest positions
     * @param alpha Weight for rigid transformation [0,1] (default: 1.0)
     * @param beta Weight for linear deformation [0,1] (default: 0.0)  
     * @param gamma Weight for quadratic deformation [0,1] (default: 0.0)
     * @return MatchResult containing rotation, translation, and goal positions
     * @note Weights are auto-normalized to sum=1. Based on Müller et al. "Meshless Deformations Based on Shape Matching"
     */
    MatchResult computeRigidTransformation(const std::vector<Particle>& particles, 
                                         float alpha = 1.0f, float beta = 0.0f, float gamma = 0.0f);
    
private:
    Eigen::Matrix3d Aqq_inverse_linear;        // A_qq^(-1) for linear (3×3)
    Eigen::Matrix<double, 9, 9> Aqq_inverse_quadratic; // A_qq^(-1) for quadratic (9×9)
    bool Aqq_linear_computed = false;
    bool Aqq_quadratic_computed = false;
    
    std::vector<Eigen::Matrix<double, 9, 1>> centeredExtendedRest; 
    Eigen::Matrix<double, 9, 1> extendedRestCentroid;              
    bool extendedRestPrecomputed = false;                          
    
    /**
     * @brief Computes weighted centroid of 3D point set
     * @param points Vector of 3D positions
     * @param weights Per-point weights (typically masses)
     * @return Weighted centroid position
     */
    Eigen::Vector3d computeCentroid(const std::vector<Eigen::Vector3d>& points,
                                   const std::vector<double>& weights);
    
    /**
     * @brief Builds 3×3 covariance matrix S = XWY^T for rigid shape matching
     * @param P Current positions (centered)
     * @param Q Rest positions (centered)
     * @param weights Per-point weights
     * @return 3×3 covariance matrix A_pq for SVD decomposition
     */
    Eigen::Matrix3d computeCovarianceMatrix(const std::vector<Eigen::Vector3d>& P,
                                           const std::vector<Eigen::Vector3d>& Q,
                                           const std::vector<double>& weights);
    
    /**
     * @brief Extracts rotation matrix from SVD, handling reflection cases
     * @param svd Completed SVD decomposition of covariance matrix
     * @return Proper rotation matrix (det=+1) using Kabsch algorithm
     */
    Eigen::Matrix3d extractRotation(const Eigen::JacobiSVD<Eigen::Matrix3d>& svd);
    
    /**
     * @brief Computes weighted centroid of 9D extended point set
     * @param points Vector of 9D extended positions [x,y,z,x²,y²,z²,xy,yz,zx]
     * @param weights Per-point weights
     * @return 9D weighted centroid
     */
    Eigen::Matrix<double, 9, 1> computeExtendedCentroid(const std::vector<Eigen::Matrix<double, 9, 1>>& points,
                                                       const std::vector<double>& weights);
    
    /**
     * @brief Builds 9×9 covariance matrix for quadratic deformations
     * @param P Current extended positions (centered)
     * @param Q Rest extended positions (centered)
     * @param weights Per-point weights
     * @return 9×9 covariance matrix for quadratic shape matching
     */
    Eigen::Matrix<double, 9, 9> computeExtendedCovarianceMatrix(const std::vector<Eigen::Matrix<double, 9, 1>>& P,
                                                               const std::vector<Eigen::Matrix<double, 9, 1>>& Q,
                                                               const std::vector<double>& weights);
    
    /**
     * @brief Computes 3×9 cross-covariance matrix between 3D current and 9D rest
     * @param current3D Current 3D positions (centered)
     * @param rest9D Rest 9D extended positions (centered)
     * @param weights Per-point weights
     * @return 3×9 matrix for quadratic deformation mapping
     */
    Eigen::Matrix<double, 3, 9> computeQuadraticCovarianceMatrix(const std::vector<Eigen::Vector3d>& current3D,
                                                                  const std::vector<Eigen::Matrix<double, 9, 1>>& rest9D,
                                                                  const std::vector<double>& weights);
    
    /**
     * @brief Extracts 3D goal positions from 9D quadratic transformation
     * @param particles Particle set with current and rest states
     * @param quadraticTransform 9×9 transformation matrix for quadratic deformation
     * @return Vector of 3D goal positions for each particle
     */
    std::vector<Eigen::Vector3d> extractGoalPositionsFromQuadratic(
        const std::vector<Particle>& particles,
        const Eigen::Matrix<double, 9, 9>& quadraticTransform);
    
    /**
     * @brief Pre-computes static extended rest position data for performance
     * @param particles Particle set to analyze
     * @param weights Per-particle weights
     * @note Performance optimization: avoids recomputing static rest data each frame
     */
    void precomputeExtendedRestData(const std::vector<Particle>& particles,
                                   const std::vector<double>& weights);
};