#pragma once

#include <vector>
#include <memory>
#include <string>
#include <Eigen/Dense>
#include "DeformableObject.h"
#include "PhysicsEngine.h"
#include "SimulationParameters.h"

/**
 * @brief Main application class managing simulation lifecycle and user interaction
 * Handles Polyscope integration, UI rendering, and physics simulation coordination
 */
class Application {
public:
    /**
     * @brief Constructs application with default settings
     * Initializes physics engine and simulation parameters
     */
    Application();
    ~Application() = default;
    
    /**
     * @brief Initializes Polyscope, creates room, and sets up callbacks
     * Must be called before run() to prepare visualization system
     */
    void initialize();
    
    /**
     * @brief Starts main application loop with Polyscope rendering
     * Blocks until application is closed by user
     */
    void run();
    
    /**
     * @brief Cleans up resources and shuts down Polyscope
     * Called automatically on application exit
     */
    void shutdown();
    
    /**
     * @brief Resets all objects to initial positions and clears velocities
     * Restores simulation to clean starting state
     */
    void resetSimulation();
    
    /**
     * @brief Performs one physics simulation step if not paused
     * Updates positions, handles collisions, and refreshes visualization
     */
    void updateSimulation();
    
    /**
     * @brief Loads currently selected OBJ file into simulation
     * Places object at random position with default scaling
     */
    void loadSelectedModel();
    
    /**
     * @brief Adds Stanford bunny model to simulation for testing
     * Convenient method for quickly adding complex geometry
     */
    void addBunny();
    
    /**
     * @brief Removes all deformable objects from simulation
     * Clears object vector and updates visualization
     */
    void clearAllObjects();
    
    /**
     * @brief Scans data directory for available OBJ files
     * Populates file list for model selection interface
     */
    void scanForObjFiles();
    
    /**
     * @brief Renders ImGui file browser window for model selection
     * Shows available OBJ files with preview and load options
     */
    void renderFileBrowser();
    
    /**
     * @brief Handles mouse clicks for particle selection and force application
     * Casts rays to find closest particles and applies forces based on mouse movement
     */
    void handleMouseInteraction();
    
    /**
     * @brief Updates visualization of interaction ray from camera to mouse position
     * Shows ray casting for debugging particle selection
     */
    void updateRayVisualization();
    
    /**
     * @brief Renders complete ImGui interface with simulation controls
     * Includes parameter sliders, buttons, statistics, and file browser
     */
    void renderUI();
    
    /**
     * @brief Creates 3D room geometry with walls, floor, and ceiling
     * Registers room meshes with Polyscope for visualization
     */
    void createRoom();
    
private:
    std::vector<std::unique_ptr<DeformableObject>> objects_;
    PhysicsEngine physicsEngine_;
    SimulationParameters params_;
    
    bool hasAnchor_;
    size_t anchorObjectIdx_;
    size_t anchorParticleIdx_;
    Eigen::Vector3d anchorPos_;
    bool showInteractionRay_;
    
    bool showFileBrowser_;
    std::string currentDirectory_;
    std::string selectedFile_;
    std::vector<std::string> availableObjFiles_;
    int selectedFileIndex_;
    
    /**
     * @brief Main simulation update callback called by Polyscope each frame
     * Handles simulation stepping and visualization updates
     */
    void mainCallback();
    
    /**
     * @brief UI rendering callback called by Polyscope for ImGui interface
     * Renders all UI panels and handles parameter updates
     */
    void uiCallback();
    
    /**
     * @brief Static wrapper for main callback to interface with Polyscope C API
     * Forwards calls to active application instance
     */
    static void staticMainCallback();
    
    static Application* instance_; 
};