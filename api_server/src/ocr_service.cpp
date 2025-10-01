#include "ocr_service.hpp"
#include <chrono>
#include <regex>
#include <sstream>
#include <random>
#include <filesystem>
#include <drogon/drogon.h>

OCRService::OCRService(const std::string& models_directory) 
    : models_directory_(models_directory) {
    
    LOG_INFO << "Initializing OCR Service with models from: " << models_directory;
    
    // Initialize your existing GenOCR pipeline
    pipeline_ = std::make_unique<GenOCR>(models_directory, false);
    
    LOG_INFO << "OCR Service initialized successfully";
}

OCRResult OCRService::processImageFile(const std::string& image_path, const std::string& request_id) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::string req_id = request_id.empty() ? generateRequestId() : request_id;
    
    try {
        LOG_DEBUG << "Processing image: " << image_path << " [" << req_id << "]";
        
        // Use your existing Run method - it's already thread-safe
        std::string raw_output = pipeline_->Run(image_path);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        int processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
            
        LOG_DEBUG << "OCR processing completed in " << processing_time << "ms [" << req_id << "]";
        
        return parseGenOCROutput(raw_output, processing_time, req_id);
        
    } catch (const std::exception& e) {
        auto end_time = std::chrono::high_resolution_clock::now();
        int processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
            
        LOG_ERROR << "OCR processing failed: " << e.what() << " [" << req_id << "]";
        
        OCRResult result;
        result.success = false;
        result.error_message = e.what();
        result.processing_time_ms = processing_time;
        result.request_id = req_id;
        return result;
    }
}

OCRResult OCRService::processImageData(const cv::Mat& image, const std::string& request_id) {
    std::string temp_path = saveTemporaryImage(image);
    OCRResult result = processImageFile(temp_path, request_id);
    cleanupTemporaryFile(temp_path);
    return result;
}

bool OCRService::isHealthy() const {
    return pipeline_ != nullptr;
}

OCRResult OCRService::parseGenOCROutput(const std::string& output, int processing_time, const std::string& request_id) {
    OCRResult result;
    result.processing_time_ms = processing_time;
    result.request_id = request_id;
    
    if (output.find("Error:") != std::string::npos) {
        result.success = false;
        result.error_message = output;
        return result;
    }
    
    result.success = true;
    
    // Parse your existing format - extract text after "--- Final Corrected Text ---"
    size_t final_text_pos = output.find("--- Final Corrected Text ---");
    if (final_text_pos != std::string::npos) {
        size_t start = output.find('\n', final_text_pos) + 1;
        if (start < output.length()) {
            result.text = output.substr(start);
            
            // Trim whitespace
            result.text.erase(result.text.find_last_not_of(" \t\n\r") + 1);
        }
    } else {
        // Fallback - use entire output if pattern not found
        result.text = output;
    }
    
    // Parse detected objects
    size_t context_pos = output.find("Detected Context Objects:");
    if (context_pos != std::string::npos) {
        std::istringstream iss(output.substr(context_pos));
        std::string line;
        while (std::getline(iss, line)) {
            if (line.find("- ") == 0) {
                std::string obj = line.substr(2);
                // Trim whitespace
                obj.erase(obj.find_last_not_of(" \t\n\r") + 1);
                if (!obj.empty()) {
                    result.detected_objects.push_back(obj);
                }
            }
        }
    }
    
    return result;
}

std::string OCRService::generateRequestId() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(100000, 999999);
    
    auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    return "req_" + std::to_string(timestamp) + "_" + std::to_string(dis(gen));
}

std::string OCRService::saveTemporaryImage(const cv::Mat& image) {
    std::string temp_path = "./uploads/temp_" + generateRequestId() + ".png";
    cv::imwrite(temp_path, image);
    return temp_path;
}

void OCRService::cleanupTemporaryFile(const std::string& filepath) {
    try {
        std::filesystem::remove(filepath);
    } catch (const std::exception& e) {
        LOG_WARN << "Failed to cleanup temporary file " << filepath << ": " << e.what();
    }
}
