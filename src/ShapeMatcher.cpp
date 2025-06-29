#include "ShapeMatcher.h"
#include "SIMDOptimizations.h"
#include <Eigen/SVD>
#include <cmath>

/**
 * @brief Computes optimal shape transformation using blended deformation modes
 * @param particles Vector of particles with current and rest positions
 * @param alpha Weight for rigid transformation [0,1] (default: 1.0)
 * @param beta Weight for linear deformation [0,1] (default: 0.0)  
 * @param gamma Weight for quadratic deformation [0,1] (default: 0.0)
 * @return MatchResult containing rotation, translation, and goal positions
 * @note Weights are auto-normalized to sum=1. Based on Müller et al. "Meshless Deformations Based on Shape Matching"
 */
ShapeMatcher::MatchResult ShapeMatcher::computeRigidTransformation(
    const std::vector<Particle>& particles, float alpha, float beta, float gamma) {
    
    MatchResult result;
    const size_t n = particles.size();
    
    float total = alpha + beta + gamma;
    if (total > 0.0f) {
        alpha /= total;
        beta /= total; 
        gamma /= total;
    } else {
        alpha = 1.0f; beta = 0.0f; gamma = 0.0f; // Default to rigid
    }
    
    std::vector<Eigen::Vector3d> P(n), Q(n);
    std::vector<double> weights(n);
    
    for (size_t i = 0; i < n; ++i) {
        P[i] = particles[i].restPosition;
        Q[i] = particles[i].position;
        weights[i] = particles[i].mass;
    }
    
    // 1. Find center points of both current and rest configurations
    Eigen::Vector3d p_bar = computeCentroid(P, weights);
    Eigen::Vector3d q_bar = computeCentroid(Q, weights);
    
    // 2. Move both point sets to their respective centers
    std::vector<Eigen::Vector3d> x(n), y(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = P[i] - p_bar;
        y[i] = Q[i] - q_bar;
    }
    
    // 3. Set up linear deformation matrix if needed
    if (!Aqq_linear_computed && (beta > 0.0f)) {
        Eigen::Matrix3d Aqq_linear = Eigen::Matrix3d::Zero();
        for (size_t i = 0; i < n; ++i) {
            Aqq_linear += weights[i] * x[i] * x[i].transpose();
        }
        Aqq_inverse_linear = Aqq_linear.inverse();
        Aqq_linear_computed = true;
    }
    
    // 4. Prepare quadratic deformation data if needed
    if (!Aqq_quadratic_computed && (gamma > 0.0f)) {
        if (!extendedRestPrecomputed) {
            precomputeExtendedRestData(particles, weights);
        }
        
        Eigen::Matrix<double, 9, 9> Aqq_quadratic = Eigen::Matrix<double, 9, 9>::Zero();
        for (size_t i = 0; i < n; ++i) {
            Aqq_quadratic += weights[i] * centeredExtendedRest[i] * centeredExtendedRest[i].transpose();
        }
        Aqq_inverse_quadratic = Aqq_quadratic.inverse();
        Aqq_quadratic_computed = true;
    }
    
    // 5. Build correlation matrix between current and rest positions
    Eigen::Matrix3d S = computeCovarianceMatrix(x, y, weights);
    
    // 6. Decompose correlation matrix to find best rotation
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(S, Eigen::ComputeFullU | Eigen::ComputeFullV);
    
    // 7. Extract rotation while avoiding reflections
    result.rotation = extractRotation(svd);
    
    // 8. Calculate linear deformation matrix
    Eigen::Matrix3d A_linear = Eigen::Matrix3d::Identity();
    if (beta > 0.0f) {
        A_linear = S * Aqq_inverse_linear;
    }
    
    // 9. Calculate quadratic deformation matrix
    Eigen::Matrix<double, 3, 9> A_quadratic_3x9;
    if (gamma > 0.0f) {
        if (!extendedRestPrecomputed) {
            precomputeExtendedRestData(particles, weights);
        }
        
        Eigen::Matrix<double, 3, 9> A_pq_quadratic = computeQuadraticCovarianceMatrix(y, centeredExtendedRest, weights);
        A_quadratic_3x9 = A_pq_quadratic * Aqq_inverse_quadratic;
    }
    
    // 10. Calculate where each particle should move to
    result.goalPositions.resize(n);
    
    for (size_t i = 0; i < n; ++i) {
        // 10a. Calculate rigid transformation goal
        Eigen::Vector3d g_rigid = result.rotation * P[i] + (q_bar - result.rotation * p_bar);
        
        // 10b. Calculate linear deformation goal
        Eigen::Vector3d g_linear = g_rigid;
        if (beta > 0.0f) {
            g_linear = A_linear * P[i] + (q_bar - A_linear * p_bar);
        }
        
        // 10c. Calculate quadratic deformation goal
        Eigen::Vector3d g_quadratic = g_rigid;
        if (gamma > 0.0f) {
            g_quadratic = A_quadratic_3x9 * centeredExtendedRest[i] + q_bar;
        }
        
        // 10d. Combine all deformation modes based on their weights
        result.goalPositions[i] = alpha * g_rigid + beta * g_linear + gamma * g_quadratic;
    }
    
    // 11. Set translation (goal positions already computed above)
    result.translation = q_bar - result.rotation * p_bar;
    
    return result;
}

/**
 * @brief Computes weighted centroid of 3D point set
 * @param points Vector of 3D positions
 * @param weights Per-point weights (typically masses)
 * @return Weighted centroid position
 */
Eigen::Vector3d ShapeMatcher::computeCentroid(const std::vector<Eigen::Vector3d>& points,
                                             const std::vector<double>& weights) {
    // 1. Use fast SIMD computation when possible
    if (SIMDOptimizations::isAVX2Available() && points.size() >= 4) {
        return SIMDOptimizations::computeCentroid_AVX2(points, weights);
    }
    
    // 2. Calculate weighted average manually
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    double totalWeight = 0.0;
    
    for (size_t i = 0; i < points.size(); ++i) {
        centroid += weights[i] * points[i];
        totalWeight += weights[i];
    }
    
    return centroid / totalWeight;
}

/**
 * @brief Builds 3×3 covariance matrix S = XWY^T for rigid shape matching
 * @param P Current positions (centered)
 * @param Q Rest positions (centered)
 * @param weights Per-point weights
 * @return 3×3 covariance matrix A_pq for SVD decomposition
 */
Eigen::Matrix3d ShapeMatcher::computeCovarianceMatrix(
    const std::vector<Eigen::Vector3d>& P,
    const std::vector<Eigen::Vector3d>& Q,
    const std::vector<double>& weights) {
    
    // 1. Organize rest positions into matrix columns
    Eigen::Matrix3Xd X(3, P.size());
    Eigen::Matrix3Xd Y(3, Q.size());
    
    for (size_t i = 0; i < P.size(); ++i) {
        X.col(i) = P[i];
        Y.col(i) = Q[i];
    }
    
    // 2. Set up weight values for each particle
    Eigen::VectorXd W(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        W(i) = weights[i];
    }
    
    // 3. Calculate weighted correlation between position sets
    Eigen::Matrix3d S = X * W.asDiagonal() * Y.transpose();
    
    return S;
}

/**
 * @brief Extracts rotation matrix from SVD, handling reflection cases
 * @param svd Completed SVD decomposition of covariance matrix
 * @return Proper rotation matrix (det=+1) using Kabsch algorithm
 */
Eigen::Matrix3d ShapeMatcher::extractRotation(const Eigen::JacobiSVD<Eigen::Matrix3d>& svd) {
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();
    
    // 1. Test if result would be a reflection instead of rotation
    double det = (V * U.transpose()).determinant();
    
    if (det < 0) {
        // 2. Fix reflection by flipping one axis
        Eigen::Matrix3d correction = Eigen::Matrix3d::Identity();
        correction(2, 2) = -1;
        return V * correction * U.transpose();
    } else {
        // 3. Use direct result for pure rotation
        return V * U.transpose();
    }
}

/**
 * @brief Computes weighted centroid of 9D extended point set
 * @param points Vector of 9D extended positions [x,y,z,x²,y²,z²,xy,yz,zx]
 * @param weights Per-point weights
 * @return 9D weighted centroid
 */
Eigen::Matrix<double, 9, 1> ShapeMatcher::computeExtendedCentroid(
    const std::vector<Eigen::Matrix<double, 9, 1>>& points,
    const std::vector<double>& weights) {
    
    Eigen::Matrix<double, 9, 1> centroid = Eigen::Matrix<double, 9, 1>::Zero();
    double totalWeight = 0.0;
    
    for (size_t i = 0; i < points.size(); ++i) {
        centroid += weights[i] * points[i];
        totalWeight += weights[i];
    }
    
    return centroid / totalWeight;
}

/**
 * @brief Builds 9×9 covariance matrix for quadratic deformations
 * @param P Current extended positions (centered)
 * @param Q Rest extended positions (centered)
 * @param weights Per-point weights
 * @return 9×9 covariance matrix for quadratic shape matching
 */
Eigen::Matrix<double, 9, 9> ShapeMatcher::computeExtendedCovarianceMatrix(
    const std::vector<Eigen::Matrix<double, 9, 1>>& P,
    const std::vector<Eigen::Matrix<double, 9, 1>>& Q,
    const std::vector<double>& weights) {
    
    Eigen::Matrix<double, 9, 9> S = Eigen::Matrix<double, 9, 9>::Zero();
    
    for (size_t i = 0; i < P.size(); ++i) {
        S += weights[i] * P[i] * Q[i].transpose();
    }
    
    return S;
}

/**
 * @brief Computes 3×9 cross-covariance matrix between 3D current and 9D rest
 * @param current3D Current 3D positions (centered)
 * @param rest9D Rest 9D extended positions (centered)
 * @param weights Per-point weights
 * @return 3×9 matrix for quadratic deformation mapping
 */
Eigen::Matrix<double, 3, 9> ShapeMatcher::computeQuadraticCovarianceMatrix(
    const std::vector<Eigen::Vector3d>& current3D,
    const std::vector<Eigen::Matrix<double, 9, 1>>& rest9D,
    const std::vector<double>& weights) {
    
    Eigen::Matrix<double, 3, 9> A_pq = Eigen::Matrix<double, 3, 9>::Zero();
    
    for (size_t i = 0; i < current3D.size(); ++i) {
        // 1.  current3D[i] is 3D centered current position (y_i)
        // 1.a rest9D[i] is 9D centered rest position (x̃_i)
        A_pq += weights[i] * current3D[i] * rest9D[i].transpose();
    }
    
    return A_pq;
}

/**
 * @brief Extracts 3D goal positions from 9D quadratic transformation
 * @param particles Particle set with current and rest states
 * @param quadraticTransform 9×9 transformation matrix for quadratic deformation
 * @return Vector of 3D goal positions for each particle
 */
std::vector<Eigen::Vector3d> ShapeMatcher::extractGoalPositionsFromQuadratic(
    const std::vector<Particle>& particles,
    const Eigen::Matrix<double, 9, 9>& quadraticTransform) {
    
    std::vector<Eigen::Vector3d> goalPositions(particles.size());
    
    for (size_t i = 0; i < particles.size(); ++i) {
        // 1. Apply 9×9 transformation to extended rest position
        Eigen::Matrix<double, 9, 1> transformedExtended = quadraticTransform * particles[i].extendedRestPosition;
        
        // 2. Extract first 3 components as 3D position
        goalPositions[i] = transformedExtended.head<3>();
    }
    
    return goalPositions;
}

/**
 * @brief Pre-computes static extended rest position data for performance
 * @param particles Particle set to analyze
 * @param weights Per-particle weights
 * @note Performance optimization: avoids recomputing static rest data each frame
 */
void ShapeMatcher::precomputeExtendedRestData(const std::vector<Particle>& particles,
                                             const std::vector<double>& weights) {
    const size_t n = particles.size();
    
    // 1. Gather all extended rest position data
    std::vector<Eigen::Matrix<double, 9, 1>> extendedRest(n);
    for (size_t i = 0; i < n; ++i) {
        extendedRest[i] = particles[i].extendedRestPosition;
    }
    
    // 2. Find center point of extended rest positions
    extendedRestCentroid = computeExtendedCentroid(extendedRest, weights);
    
    // 3. Store centered versions for faster computation later
    centeredExtendedRest.resize(n);
    for (size_t i = 0; i < n; ++i) {
        centeredExtendedRest[i] = extendedRest[i] - extendedRestCentroid;
    }
    
    extendedRestPrecomputed = true;
}

