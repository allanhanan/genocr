#include <drogon/drogon.h>
#include <iostream>
#include <filesystem>

int main() {
    try {
        std::cout << "Starting GenOCR API Server..." << std::endl;
        
        // Check if models directory exists
        if (!std::filesystem::exists("../../models")) {
            std::cerr << "Warning: Models directory not found at ../../models" << std::endl;
        }
        
        // Create uploads directory if it doesn't exist
        std::filesystem::create_directories("./uploads");
        
        // Load configuration
        drogon::app().loadConfigFile("./config.json");
        
        std::cout << "GenOCR API Server starting on port 8080..." << std::endl;
        std::cout << "Endpoints:" << std::endl;
        std::cout << "  POST /api/v1/ocr/process - Process OCR request" << std::endl;
        std::cout << "  GET  /api/v1/health      - Health check" << std::endl;
        std::cout << "  GET  /api/v1/metrics     - Server metrics" << std::endl;
        
        // Run the server
        drogon::app().run();
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to start server: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
