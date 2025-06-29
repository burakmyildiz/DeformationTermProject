#include "Application.h"

/**
 * @brief Main entry point for meshless deformation simulation
 * @param argc Command line argument count (unused)
 * @param argv Command line arguments (unused)
 * @return Exit status (0 for success)
 */
int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    
    Application app;
    
    app.initialize();
    app.run();
    app.shutdown();
    
    return 0;
}