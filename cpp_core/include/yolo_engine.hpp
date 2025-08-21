#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

/**
 * @class YoloEngine
 * @brief Manages the YOLOv8 model for object context detection.
 */
class YoloEngine {
public:
    YoloEngine(Ort::Env& env, Ort::SessionOptions& session_options, const std::string& model_path);
    std::vector<std::string> Detect(const cv::Mat& image);

private:
    std::unique_ptr<Ort::Session> session_;
    
    std::vector<std::string> input_names_str_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::string> class_names_;
};