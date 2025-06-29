#pragma once

/**
 * @brief Central configuration structure for physics simulation parameters
 * Contains all tunable parameters for simulation behavior, performance, and visualization
 */
struct SimulationParameters {
    float timeStep = 0.012f;         // Time step duration for integration

    float gravity = -9.81f;          
    float damping = 0.98f;           // Velocity damping factor to prevent oscillation [0,1]
    float stiffness = 0.4f;          // Shape matching stiffness [0,1]
    float particleMass = 1.0f;       // Mass of each particle in kg

    float groundHeight = -3.0f;      // Y-coordinate of ground plane
    float restitution = 0.2f;        //Collision elasticity [0=inelastic, 1=elastic]
    float particleRadius = 0.25f;    // Particle collision radius
    
    // Room boundaries 
    float roomSize = 20.0f;          // -roomSize to +roomSize in X and Z
    float roomHeight = 15.0f;        // Room height above ground 
    
    float separationStrength = 1.2f; // Collision separation force multiplier
    float maxVelocity = 8.0f;        
    float maxPositionCorrection = 0.05f; 
    
    float mouseForceRadius = 0.8f;   
    float mouseForceStrength = 100.0f; 
    
    // Deformation params - 3-way blending system
    float alpha = 1.0f;              // Rigid transformation weight [0,1]
    float beta = 0.0f;               // Linear deformation weight [0,1]
    float gamma = 0.0f;              // Quadratic deformation weight [0,1]
                                     //  alpha + beta + gamma = 1 
    
    float friction = 0.95f;              // Surface friction coefficient [0,1]
    float wallFriction = 0.95f;          // Wall friction coefficient [0,1] 
    float adaptiveStiffnessTau = 0.03f;
    float groundStickiness = 0.8f;       
    float contactOffset = 0.85f;           
    
    float spatialHashCellSize = 0.6f;    // Cell size for spatial hashing grid
    bool useSpatialHashing = true;       
    bool useOctree = false;              
    
    int maxCollisionPairsPerParticle = 8;   
    float collisionCullDistance = 0.5f;      
    bool useAdaptiveIterations = true;       
    
    bool isPaused = false;        
    bool showParticles = true;    
    bool showMesh = true;         
    bool showGroundPlane = true;  
};