#pragma once
#include "paddle_utils.hpp"
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

/**
 * @class PaddleOcrEngine
 * @brief Manages the PaddleOCR models for text extraction.
 */
class PaddleOcrEngine {
public:
    PaddleOcrEngine(Ort::Env& env, Ort::SessionOptions& session_options, const std::string& models_dir);
    std::vector<PaddleUtils::OcrResult> ExtractText(const cv::Mat& image);

private:
    void LoadCharset(const std::string& path);

    std::unique_ptr<Ort::Session> det_session_;
    std::unique_ptr<Ort::Session> rec_session_;
    
    std::vector<std::string> det_input_names_str_;
    std::vector<std::string> det_output_names_str_;
    std::vector<const char*> det_input_names_;
    std::vector<const char*> det_output_names_;

    std::vector<std::string> rec_input_names_str_;
    std::vector<std::string> rec_output_names_str_;
    std::vector<const char*> rec_input_names_;
    std::vector<const char*> rec_output_names_;
    
    std::vector<std::string> charset_;
};