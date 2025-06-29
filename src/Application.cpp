#include "Application.h"
#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <polyscope/point_cloud.h>
#include <polyscope/curve_network.h>
#include <polyscope/view.h>
#include <polyscope/pick.h>
#include "imgui.h"
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include "SIMDOptimizations.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

Application* Application::instance_ = nullptr;

/**
 * @brief Constructs application with default settings
 * Initializes physics engine and simulation parameters
 */
Application::Application() 
    : hasAnchor_(false), anchorObjectIdx_(-1), anchorParticleIdx_(-1),
      showInteractionRay_(false), showFileBrowser_(false),
      currentDirectory_("../data"), selectedFileIndex_(-1) {
    instance_ = this;
}

/**
 * @brief Initializes Polyscope, creates room, and sets up callbacks
 * Must be called before run() to prepare visualization system
 */
void Application::initialize() {
    // 1. Configure and start Polyscope visualization
    polyscope::options::programName = "Shape Matching Deformation";
    polyscope::options::verbosity = 0;
    polyscope::init();
    
    // 2. Display performance capabilities to console
    std::cout << "=== Performance Optimization Status ===" << std::endl;
    std::cout << "AVX2 Support: " << (SIMDOptimizations::isAVX2Available() ? "✓ ENABLED" : "✗ DISABLED") << std::endl;
    std::cout << "SIMD Optimizations: " << (SIMDOptimizations::isAVX2Available() ? "Active" : "Fallback to scalar") << std::endl;
    
#ifdef USE_OPENMP
    std::cout << "OpenMP Support: ✓ ENABLED" << std::endl;
    std::cout << "OpenMP Threads: " << omp_get_max_threads() << std::endl;
#else
    std::cout << "OpenMP Support: ✗ DISABLED" << std::endl;
#endif
    
    std::cout << "=========================================" << std::endl;
    
    // 3. Position camera for good view of simulation area
    polyscope::view::lookAt(
        glm::vec3(5.0f, 3.0f, 8.0f),
        glm::vec3(0.0f, 2.0f, 0.0f)
    );
    
    // 4. Set up simulation environment
    resetSimulation();
    createRoom();
    physicsEngine_.setParameters(params_);
    
    // 5. Load default test object
    auto bunny = std::make_unique<DeformableObject>("bunny");
    if (bunny->loadFromOBJ("../data/bunny.obj", Eigen::Vector3d(0, 3, 0), 10.0)) {
        bunny->initializeVisualization();
        objects_.push_back(std::move(bunny));
    }
    
    // 6. Connect UI callback system
    polyscope::state::userCallback = staticMainCallback;
}

/**
 * @brief Starts main application loop with Polyscope rendering
 * Blocks until application is closed by user
 */
void Application::run() {
    polyscope::show();
}

/**
 * @brief Cleans up resources and shuts down Polyscope
 * Called automatically on application exit
 */
void Application::shutdown() {
    instance_ = nullptr;
}

/**
 * @brief Resets all objects to initial positions and clears velocities
 * Restores simulation to clean starting state
 */
void Application::resetSimulation() {
    hasAnchor_ = false;
    for (auto& obj : objects_) {
        if(obj) obj->reset();
    }
}

/**
 * @brief Creates 3D room geometry with walls, floor, and ceiling
 * Registers room meshes with Polyscope for visualization
 */
void Application::createRoom() {
    double roomSize = 20.0;
    double roomHeight = 15.0;
    double groundLevel = params_.groundHeight;
    
    // 1. Create floor surface
    std::vector<Eigen::Vector3d> floorVertices = {
        {-roomSize, groundLevel, -roomSize}, 
        { roomSize, groundLevel, -roomSize},
        { roomSize, groundLevel,  roomSize}, 
        {-roomSize, groundLevel,  roomSize}
    };
    std::vector<std::array<size_t, 3>> floorFaces = {{0, 1, 2}, {0, 2, 3}};
    auto floor = polyscope::registerSurfaceMesh("floor", floorVertices, floorFaces);
    floor->setSurfaceColor({0.6, 0.5, 0.4});
    
    // 2. Build back wall for depth reference
    std::vector<Eigen::Vector3d> backWallVertices = {
        {-roomSize, groundLevel, -roomSize},
        { roomSize, groundLevel, -roomSize},
        { roomSize, groundLevel + roomHeight, -roomSize},
        {-roomSize, groundLevel + roomHeight, -roomSize}
    };
    std::vector<std::array<size_t, 3>> wallFaces = {{0, 1, 2}, {0, 2, 3}};
    auto backWall = polyscope::registerSurfaceMesh("back_wall", backWallVertices, wallFaces);
    backWall->setSurfaceColor({0.9, 0.95, 1.0});
    
    // 3. Add side walls
    std::vector<Eigen::Vector3d> leftWallVertices = {
        {-roomSize, groundLevel, -roomSize},
        {-roomSize, groundLevel + roomHeight, -roomSize},
        {-roomSize, groundLevel + roomHeight,  roomSize},
        {-roomSize, groundLevel,  roomSize}
    };
    auto leftWall = polyscope::registerSurfaceMesh("left_wall", leftWallVertices, wallFaces);
    leftWall->setSurfaceColor({1.0, 0.9, 0.9});
    
    std::vector<Eigen::Vector3d> rightWallVertices = {
        { roomSize, groundLevel, -roomSize},
        { roomSize, groundLevel,  roomSize},
        { roomSize, groundLevel + roomHeight,  roomSize},
        { roomSize, groundLevel + roomHeight, -roomSize}
    };
    auto rightWall = polyscope::registerSurfaceMesh("right_wall", rightWallVertices, wallFaces);
    rightWall->setSurfaceColor({0.9, 1.0, 0.9});
    
    // 4. Add semi-transparent ceiling
    std::vector<Eigen::Vector3d> ceilingVertices = {
        {-roomSize, groundLevel + roomHeight, -roomSize}, 
        {-roomSize, groundLevel + roomHeight,  roomSize},
        { roomSize, groundLevel + roomHeight,  roomSize}, 
        { roomSize, groundLevel + roomHeight, -roomSize}
    };
    auto ceiling = polyscope::registerSurfaceMesh("ceiling", ceilingVertices, floorFaces);
    ceiling->setSurfaceColor({1.0, 1.0, 0.95});
    ceiling->setTransparency(0.4);
}

/**
 * @brief Scans data directory for available OBJ files
 * Populates file list for model selection interface
 */
void Application::scanForObjFiles() {
    availableObjFiles_.clear();
    
    try {
        // 1. Check if data directory exists
        if (std::filesystem::exists(currentDirectory_) && std::filesystem::is_directory(currentDirectory_)) {
            // 2. Find all OBJ files in directory
            for (const auto& entry : std::filesystem::directory_iterator(currentDirectory_)) {
                if (entry.is_regular_file() && entry.path().extension() == ".obj") {
                    availableObjFiles_.push_back(entry.path().filename().string());
                }
            }
        }
        
        // 3. Sort files alphabetically for easier browsing
        std::sort(availableObjFiles_.begin(), availableObjFiles_.end());
    } catch (const std::exception& e) {
        std::cerr << "Error scanning directory: " << e.what() << std::endl;
    }
}

/**
 * @brief Renders ImGui file browser window for model selection
 * Shows available OBJ files with preview and load options
 */
void Application::renderFileBrowser() {
    if (!showFileBrowser_) return;
    
    ImGui::Begin("Load OBJ Model", &showFileBrowser_, ImGuiWindowFlags_AlwaysAutoResize);
    
    // 1. Show current directory and refresh option
    ImGui::Text("Directory: %s", currentDirectory_.c_str());
    
    if (ImGui::Button("Refresh")) {
        scanForObjFiles();
        selectedFileIndex_ = -1;
        selectedFile_ = "";
    }
    
    ImGui::Separator();
    ImGui::Text("Available OBJ Files:");
    
    // 2. Display file list or show empty message
    if (availableObjFiles_.empty()) {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "No .obj files found in directory");
    } else {
        std::vector<const char*> fileNames;
        for (const auto& file : availableObjFiles_) {
            fileNames.push_back(file.c_str());
        }
        
        if (ImGui::ListBox("##files", &selectedFileIndex_, fileNames.data(), fileNames.size(), 8)) {
            if (selectedFileIndex_ >= 0 && selectedFileIndex_ < static_cast<int>(availableObjFiles_.size())) {
                selectedFile_ = availableObjFiles_[selectedFileIndex_];
            }
        }
    }
    
    ImGui::Separator();
    
    // 3. Handle file selection and loading
    if (!selectedFile_.empty()) {
        ImGui::Text("Selected: %s", selectedFile_.c_str());
        
        if (ImGui::Button("Add Model")) {
            loadSelectedModel();
        }
        ImGui::SameLine();
    }
    
    if (ImGui::Button("Close")) {
        showFileBrowser_ = false;
        selectedFileIndex_ = -1;
        selectedFile_ = "";
    }
    
    // 4. Quick access for common test models
    ImGui::Separator();
    ImGui::Text("Quick Load (Simple Models):");
    
    if (ImGui::Button("Suzanne")) {
        selectedFile_ = "suzanne.obj";
        loadSelectedModel();
    }
    ImGui::SameLine();
    if (ImGui::Button("Sphere")) {
        selectedFile_ = "sphere.obj";
        loadSelectedModel();
    }
    
    if (ImGui::Button("Torus")) {
        selectedFile_ = "torus.obj";
        loadSelectedModel();
    }
    ImGui::SameLine();
    if (ImGui::Button("Gourd")) {
        selectedFile_ = "gourd.obj";
        loadSelectedModel();
    }
    
    if (ImGui::Button("Cow")) {
        selectedFile_ = "cow.obj";
        loadSelectedModel();
    }
    ImGui::SameLine();
    if (ImGui::Button("Teapot")) {
        selectedFile_ = "teapot.obj";
        loadSelectedModel();
    }
    
    ImGui::End();
}

/**
 * @brief Loads currently selected OBJ file into simulation
 * Places object at random position with default scaling
 */
void Application::loadSelectedModel() {
    if (selectedFile_.empty()) {
        std::cerr << "No file selected!" << std::endl;
        return;
    }
    
    // 1. Build full file path and clear interactions
    std::string fullPath = currentDirectory_ + "/" + selectedFile_;
    hasAnchor_ = false;
    
    // 2. Create unique object name and instance
    std::string baseName = selectedFile_.substr(0, selectedFile_.find_last_of('.'));
    std::string objectName = baseName + "_" + std::to_string(objects_.size());
    auto newObject = std::make_unique<DeformableObject>(objectName);
    
    // 3. Apply model-specific scaling corrections
    double scale = 1.0;
    if (selectedFile_ == "bunny.obj") {
        scale = 10.0;
    } else if (selectedFile_ == "teapot.obj") {
        scale = 0.1;
    } else if (selectedFile_ == "sphere.obj") {
        scale = 0.5;
    } else if (selectedFile_ == "torus.obj") {
        scale = 0.5;
    } else if (selectedFile_ == "gourd.obj") {
        scale = 2.0;
    } else if (selectedFile_ == "humanoid_tri.obj") {
        scale = 0.3;
    }
    
    // 4. Calculate spawn position in grid layout
    double spacing = 3.0;
    int numObjects = objects_.size();
    double x = (numObjects % 3 - 1) * spacing;
    double z = (numObjects / 3 - 1) * spacing;
    Eigen::Vector3d spawnPosition(x, 3, z);
    
    // 5. Load and register the new object
    if (newObject->loadFromOBJ(fullPath, spawnPosition, scale)) {
        newObject->initializeVisualization();
        objects_.push_back(std::move(newObject));
        std::cout << "Successfully loaded: " << selectedFile_ << " with scale: " << scale << std::endl;
    } else {
        std::cerr << "Failed to load: " << selectedFile_ << std::endl;
    }
}

/**
 * @brief Adds Stanford bunny model to simulation for testing
 * Convenient method for quickly adding complex geometry
 */
void Application::addBunny() {
    // 1. Create bunny object with unique name
    auto bunny = std::make_unique<DeformableObject>("bunny_" + std::to_string(objects_.size()));
    
    // 2. Position using grid layout system
    double spacing = 3.0;
    int numObjects = objects_.size();
    double x = (numObjects % 3 - 1) * spacing;
    double z = (numObjects / 3 - 1) * spacing;
    
    // 3. Load and add to simulation
    if (bunny->loadFromOBJ("../data/bunny.obj", Eigen::Vector3d(x, 4, z), 10.0)) {
        bunny->initializeVisualization();
        objects_.push_back(std::move(bunny));
    }
}

/**
 * @brief Removes all deformable objects from simulation
 * Clears object vector and updates visualization
 */
void Application::clearAllObjects() {
    // 1. Clear all interactions and objects
    hasAnchor_ = false;
    objects_.clear();
    
    // 2. Reset visualization and rebuild room
    polyscope::removeAllStructures();
    createRoom();
}

// Static callback wrapper
/**
 * @brief Static wrapper for main callback to interface with Polyscope C API
 * Forwards calls to active application instance
 */
void Application::staticMainCallback() {
    if (instance_) {
        instance_->mainCallback();
    }
}

/**
 * @brief Main simulation update callback called by Polyscope each frame
 * Handles simulation stepping and visualization updates
 */
void Application::mainCallback() {
    uiCallback();
    renderFileBrowser();
    updateSimulation();
}

/**
 * @brief Handles mouse clicks for particle selection and force application
 * Casts rays to find closest particles and applies forces based on mouse movement
 */
void Application::handleMouseInteraction() {
    if (ImGui::GetIO().WantCaptureMouse) {
        return;
    }

    bool leftMousePressed = ImGui::IsMouseDown(ImGuiMouseButton_Left);
    bool rightMousePressed = ImGui::IsMouseDown(ImGuiMouseButton_Right);
    
    // 1. Right click clears any active interactions
    if (rightMousePressed) {
        hasAnchor_ = false;
        if (polyscope::hasPointCloud("anchor_point")) {
            polyscope::removePointCloud("anchor_point");
        }
        if (polyscope::hasPointCloud("cursor_point")) {
            polyscope::removePointCloud("cursor_point");
        }
        if (polyscope::hasCurveNetwork("interaction_line")) {
            polyscope::removeCurveNetwork("interaction_line");
        }
        return;
    }
    
    // 2. Left click selects a particle to interact with
    if (leftMousePressed && !hasAnchor_) {
        // 2a. Get ray from camera through mouse position
        ImVec2 mousePos = ImGui::GetMousePos();
        glm::vec2 screenCoords(mousePos.x, mousePos.y);
        glm::vec3 rayDir = polyscope::view::screenCoordsToWorldRay(screenCoords);
        glm::vec3 rayOrigin = polyscope::view::getCameraWorldPosition();
        
        double minDepth = std::numeric_limits<double>::infinity();
        bool foundParticle = false;
        
        // 2b. Find closest particle intersecting the mouse ray
        for (size_t i = 0; i < objects_.size(); ++i) {
            if (!objects_[i]) continue;
            auto& particles = objects_[i]->getParticles();
            
            for (size_t j = 0; j < particles.size(); ++j) {
                Eigen::Vector3d pPos = particles[j].position;
                glm::vec3 pGlm(pPos.x(), pPos.y(), pPos.z());
                
                glm::vec3 toParticle = pGlm - rayOrigin;
                float t = glm::dot(toParticle, rayDir);
                if (t < 0.1f) continue;
                
                glm::vec3 closestPoint = rayOrigin + t * rayDir;
                float distSq = glm::length2(pGlm - closestPoint);
                
                float selectionRadius = 0.15f;
                
                if (distSq < selectionRadius * selectionRadius && t < minDepth) {
                    minDepth = t;
                    anchorObjectIdx_ = i;
                    anchorParticleIdx_ = j;
                    anchorPos_ = pPos;
                    foundParticle = true;
                }
            }
        }
        
        // 3. Create visual anchor for selected particle
        if (foundParticle) {
            hasAnchor_ = true;
            
            std::vector<Eigen::Vector3d> anchorPosVec = {anchorPos_};
            auto anchor = polyscope::registerPointCloud("anchor_point", anchorPosVec);
            anchor->setPointColor({0.0, 0.0, 1.0});
            
            glm::vec3 cameraPos = polyscope::view::getCameraWorldPosition();
            glm::vec3 anchorGlm(anchorPos_.x(), anchorPos_.y(), anchorPos_.z());
            float camDist = glm::length(anchorGlm - cameraPos);
            anchor->setPointRadius(camDist * 0.0005f);
        }
    }
    
    // 4. Handle active interaction (dragging)
    if (hasAnchor_) {
        ImVec2 mousePos = ImGui::GetMousePos();
        glm::vec2 screenCoords(mousePos.x, mousePos.y);
        glm::vec3 rayDir = polyscope::view::screenCoordsToWorldRay(screenCoords);
        glm::vec3 rayOrigin = polyscope::view::getCameraWorldPosition();
        
        glm::vec3 anchorGlm(anchorPos_.x(), anchorPos_.y(), anchorPos_.z());
        float depth = glm::length(anchorGlm - rayOrigin);
        
        float wheel = ImGui::GetIO().MouseWheel;
        if (wheel != 0.0f) {
            depth += wheel * 0.5f;
            depth = std::max(1.0f, depth);
        }
        
        glm::vec3 cursorGlm = rayOrigin + depth * rayDir;
        Eigen::Vector3d cursorPos(cursorGlm.x, cursorGlm.y, cursorGlm.z);
        
        std::vector<Eigen::Vector3d> cursorPosVec = {cursorPos};
        
        glm::vec3 cameraPos = polyscope::view::getCameraWorldPosition();
        float camDist = glm::length(cursorGlm - cameraPos);
        float sphereRadius = camDist * 0.0005f;
        
        if (!polyscope::hasPointCloud("cursor_point")) {
            auto cursor = polyscope::registerPointCloud("cursor_point", cursorPosVec);
            cursor->setPointColor({1.0, 0.0, 0.0});
            cursor->setPointRadius(sphereRadius);
        } else {
            polyscope::getPointCloud("cursor_point")->updatePointPositions(cursorPosVec);
            polyscope::getPointCloud("cursor_point")->setPointRadius(sphereRadius);
        }
        
        std::vector<Eigen::Vector3d> linePoints = {anchorPos_, cursorPos};
        std::vector<std::array<size_t, 2>> edges = {{0, 1}};
        if (polyscope::hasCurveNetwork("interaction_line")) {
            polyscope::removeCurveNetwork("interaction_line");
        }
        auto line = polyscope::registerCurveNetwork("interaction_line", linePoints, edges);
        line->setColor({0.3, 0.3, 0.3});
        line->setRadius(0.002);
        
        if (anchorObjectIdx_ < objects_.size() && objects_[anchorObjectIdx_]) {
            auto& particles = objects_[anchorObjectIdx_]->getParticles();
            if (anchorParticleIdx_ < particles.size()) {
                auto& anchorParticle = particles[anchorParticleIdx_];
                
                anchorPos_ = anchorParticle.position;
                std::vector<Eigen::Vector3d> newAnchorPos = {anchorPos_};
                polyscope::getPointCloud("anchor_point")->updatePointPositions(newAnchorPos);
                
                glm::vec3 cameraPos = polyscope::view::getCameraWorldPosition();
                glm::vec3 anchorGlm(anchorPos_.x(), anchorPos_.y(), anchorPos_.z());
                float anchorCamDist = glm::length(anchorGlm - cameraPos);
                polyscope::getPointCloud("anchor_point")->setPointRadius(anchorCamDist * 0.0005f);
                
                Eigen::Vector3d displacement = cursorPos - anchorPos_;
                Eigen::Vector3d force = displacement * params_.mouseForceStrength;
                
                for (size_t i = 0; i < particles.size(); ++i) {
                    auto& p = particles[i];
                    Eigen::Vector3d diff = p.position - anchorPos_;
                    double dist = diff.norm();
                    
                    double falloff = exp(-(dist * dist) / (2.0 * params_.mouseForceRadius * params_.mouseForceRadius));
                    p.velocity += force * falloff * params_.timeStep;
                }
            }
        }
    }
}

/**
 * @brief Updates visualization of interaction ray from camera to mouse position
 * Shows ray casting for debugging particle selection
 */
void Application::updateRayVisualization() {
    if (!showInteractionRay_ || ImGui::GetIO().WantCaptureMouse) {
        if (polyscope::hasCurveNetwork("interaction_ray")) {
            polyscope::removeCurveNetwork("interaction_ray");
        }
        return;
    }
    
    // 1. Calculate ray from camera through mouse position
    ImVec2 mousePos = ImGui::GetMousePos();
    glm::vec2 screenCoords(mousePos.x, mousePos.y);
    glm::vec3 rayDir = polyscope::view::screenCoordsToWorldRay(screenCoords);
    glm::vec3 rayOrigin = polyscope::view::getCameraWorldPosition();
    
    // 2. Build ray geometry for visualization
    std::vector<Eigen::Vector3d> rayPoints;
    glm::vec3 rayStart = rayOrigin + 0.1f * rayDir;
    rayPoints.push_back(Eigen::Vector3d(rayStart.x, rayStart.y, rayStart.z));
    
    float rayLength = 10.0f;
    glm::vec3 rayEnd = rayOrigin + rayLength * rayDir;
    rayPoints.push_back(Eigen::Vector3d(rayEnd.x, rayEnd.y, rayEnd.z));
    
    std::vector<std::array<size_t, 2>> edges = {{0, 1}};
    
    // 3. Replace existing ray visualization
    if (polyscope::hasCurveNetwork("interaction_ray")) {
        polyscope::removeCurveNetwork("interaction_ray");
    }
    
    auto ray = polyscope::registerCurveNetwork("interaction_ray", rayPoints, edges);
    ray->setRadius(0.002);
    
    // 4. Color ray based on interaction state
    if (hasAnchor_) {
        ray->setColor({0.3, 0.3, 1.0});
    } else {
        ray->setColor({0.4, 0.4, 0.4});
    }
}

/**
 * @brief Performs one physics simulation step if not paused
 * Updates positions, handles collisions, and refreshes visualization
 */
void Application::updateSimulation() {
    // 1. Process user interactions first
    handleMouseInteraction();
    updateRayVisualization();
    
    if (!params_.isPaused) {
        // 2. Update physics for each object
        for (size_t i = 0; i < objects_.size(); ++i) {
            if (objects_[i]) {
                objects_[i]->update(physicsEngine_, params_.timeStep);
            }
        }
        
        // 3. Resolve collisions between different objects
        physicsEngine_.handleInterObjectCollisions(objects_);
    }
}

/**
 * @brief UI rendering callback called by Polyscope for ImGui interface
 * Renders all UI panels and handles parameter updates
 */
void Application::uiCallback() {
    ImGui::PushItemWidth(150);
    
    // 1. Basic simulation controls
    if (ImGui::Button("Reset")) { resetSimulation(); }
    ImGui::SameLine();
    ImGui::Checkbox("Pause", &params_.isPaused);
    
    ImGui::Separator();
    ImGui::Text("Simulation Parameters");
    ImGui::SliderFloat("Time Step", &params_.timeStep, 0.001f, 0.05f, "%.4f");
    ImGui::SliderFloat("Gravity", &params_.gravity, -20.0f, 0.0f);
    ImGui::SliderFloat("Stiffness (alpha)", &params_.stiffness, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Damping", &params_.damping, 0.8f, 1.0f, "%.3f");
    ImGui::SliderFloat("Restitution", &params_.restitution, 0.0f, 1.0f);
    ImGui::SliderFloat("Particle Mass", &params_.particleMass, 0.1f, 5.0f);
    
    // 2. Deformation mode controls
    ImGui::Separator();
    ImGui::Text("Deformation Mode Blending (Auto-normalized)");
    
    ImGui::SliderFloat("Rigid (α)", &params_.alpha, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Linear (β)", &params_.beta, 0.0f, 1.0f, "%.2f");
    ImGui::SliderFloat("Quadratic (γ)", &params_.gamma, 0.0f, 1.0f, "%.2f");
    
    float total = params_.alpha + params_.beta + params_.gamma;
    if (total > 0.0f) {
        params_.alpha /= total;
        params_.beta /= total;
        params_.gamma /= total;
    } else {
        params_.alpha = 1.0f; params_.beta = 0.0f; params_.gamma = 0.0f;
    }
    
    ImGui::Text("Normalized: α=%.2f, β=%.2f, γ=%.2f", params_.alpha, params_.beta, params_.gamma);
    
    if (ImGui::Button("Pure Rigid")) { params_.alpha = 1.0f; params_.beta = 0.0f; params_.gamma = 0.0f; }
    ImGui::SameLine();
    if (ImGui::Button("Pure Linear")) { params_.alpha = 0.0f; params_.beta = 1.0f; params_.gamma = 0.0f; }
    ImGui::SameLine();
    if (ImGui::Button("Pure Quadratic")) { params_.alpha = 0.0f; params_.beta = 0.0f; params_.gamma = 1.0f; }
    
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::Text("3-Way Deformation Blending:");
        ImGui::Text("α (Rigid): Pure shape preservation");
        ImGui::Text("β (Linear): Stretch/compress/shear");
        ImGui::Text("γ (Quadratic): Bending/twisting modes");
        ImGui::Text("Weights are auto-normalized to sum = 1");
        ImGui::EndTooltip();
    }
    
    ImGui::Separator();
    ImGui::Text("Mouse Interaction");
    ImGui::SliderFloat("Force Strength", &params_.mouseForceStrength, 10.0f, 1000.0f);
    ImGui::SliderFloat("Force Radius", &params_.mouseForceRadius, 0.5f, 5.0f);
    ImGui::Checkbox("Show Ray", &showInteractionRay_);
    if (hasAnchor_) {
        ImGui::TextColored(ImVec4(0.0f, 0.0f, 1.0f, 1.0f), "Blue: Anchor point");
        ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Red: Cursor (follows mouse)");
        ImGui::Text("Mouse wheel: Adjust cursor depth");
        ImGui::Text("Right click: Clear anchor");
    } else {
        ImGui::Text("Left click: Select anchor point");
        ImGui::Text("Right click: Clear selection");
    }

    ImGui::Separator();
    ImGui::Text("Collision");
    ImGui::SliderFloat("Particle Radius", &params_.particleRadius, 0.05f, 0.5f);
    ImGui::SliderFloat("Ground Height", &params_.groundHeight, -5.0f, 0.0f);
    
    ImGui::Separator();
    ImGui::Text("Advanced Physics");
    ImGui::SliderFloat("Friction", &params_.friction, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Wall Friction", &params_.wallFriction, 0.0f, 1.0f, "%.3f");
    ImGui::SliderFloat("Ground Stickiness", &params_.groundStickiness, 0.0f, 1.0f, "%.3f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Extra damping when restitution=0 for clay-like behavior");
    }
    ImGui::SliderFloat("Contact Offset", &params_.contactOffset, 0.5f, 1.0f, "%.3f");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Visual contact distance (lower = objects appear to touch more closely)");
    }
    ImGui::SliderFloat("Adaptive Tau", &params_.adaptiveStiffnessTau, 0.01f, 0.1f, "%.3f");
    
    ImGui::Separator();
    ImGui::Text("Optimization");
    
    if (ImGui::RadioButton("Brute Force", !params_.useSpatialHashing && !params_.useOctree)) {
        params_.useSpatialHashing = false;
        params_.useOctree = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Spatial Hash", params_.useSpatialHashing && !params_.useOctree)) {
        params_.useSpatialHashing = true;
        params_.useOctree = false;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Octree", params_.useOctree)) {
        params_.useSpatialHashing = false;
        params_.useOctree = true;
    }
    
    if (params_.useSpatialHashing) {
        ImGui::SliderFloat("Hash Cell Size", &params_.spatialHashCellSize, 0.5f, 3.0f, "%.1f");
    }
    
    ImGui::Checkbox("Adaptive Iterations", &params_.useAdaptiveIterations);
    ImGui::SliderInt("Max Collisions/Particle", &params_.maxCollisionPairsPerParticle, 4, 16);
    ImGui::SliderFloat("Collision Cull Distance", &params_.collisionCullDistance, 0.1f, 1.0f, "%.2f");
    
    ImGui::Separator();
    ImGui::Text("Model Loading");
    if (ImGui::Button("Browse Models...")) {
        scanForObjFiles();
        showFileBrowser_ = true;
    }
    
    ImGui::Separator();
    ImGui::Text("Scene Objects: %zu", objects_.size());
    
    if (ImGui::Button("Add Bunny")) {
        addBunny();
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear All")) { 
        clearAllObjects();
    }
    
    physicsEngine_.setParameters(params_);
    
    ImGui::PopItemWidth();
}