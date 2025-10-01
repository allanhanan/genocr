#pragma once

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include "genocr.hpp"

struct OCRResult {
    bool success;
    std::string text;
    std::vector<std::string> detected_objects;
    std::string error_message;
    int processing_time_ms;
    std::string request_id;
};

class OCRService {
public:
    explicit OCRService(const std::string& models_directory);
    
    // Main processing methods
    OCRResult processImageFile(const std::string& image_path, const std::string& request_id = "");
    OCRResult processImageData(const cv::Mat& image, const std::string& request_id = "");
    
    // Health check
    bool isHealthy() const;
    
    // Get service info
    std::string getVersion() const { return "1.0.0"; }

private:
    std::unique_ptr<GenOCR> pipeline_;
    std::string models_directory_;
    
    // Helper methods
    OCRResult parseGenOCROutput(const std::string& output, int processing_time, const std::string& request_id);
    std::string generateRequestId();
    std::string saveTemporaryImage(const cv::Mat& image);
    void cleanupTemporaryFile(const std::string& filepath);
};
