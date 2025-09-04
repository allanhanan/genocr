#include "genocr.hpp"
#include "paddle_utils.hpp"
#include <iostream>
#include <sstream>
#include <filesystem>

GenOCR::GenOCR(const std::string& models_directory, bool use_cuda) {
    
    env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "genocr_safe");
    
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);  
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    // NO CUDA - causes threading issues
    std::cout << "CPU-only mode for maximum stability" << std::endl;

    yolo_engine_ = std::make_unique<YoloEngine>(*env_, session_options, models_directory + "/detector.onnx");
    paddle_engine_ = std::make_unique<PaddleOcrEngine>(*env_, session_options, models_directory);  
    corrector_engine_ = std::make_unique<CorrectorEngine>(models_directory + "/corrector");
    std::cout << "All engines loaded (CPU-only, single-threaded)" << std::endl;
}

std::string GenOCR::Run(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        return "Error: Failed to load image at: " + image_path;
    }

    std::ostringstream result;
    std::string filename = std::filesystem::path(image_path).filename().string();
    result << "=== OCR Results for " << filename << " ===\n";

    try {
        std::vector<std::string> context = yolo_engine_->Detect(image);
        
        if (!context.empty()) {
            result << "\nDetected Context Objects:\n";
            for (const auto& obj : context) {
                result << "- " << obj << "\n";
            }
        }

        std::vector<PaddleUtils::OcrResult> ocr_results = paddle_engine_->ExtractText(image);
        
        result << "\n--- Raw OCR Output ---\n";
        std::string full_raw_text;
        for (const auto& ocr_result : ocr_results) {
            if (!ocr_result.text.empty()) {
                result << "RAW LINE: \"" << ocr_result.text << "\"\n";
                full_raw_text += ocr_result.text + "\n";
            }
        }

        if (full_raw_text.empty()) {
            result << "No text detected.\n";
        } else {
            std::string corrected = corrector_engine_->Correct(image, context, full_raw_text);
            result << "\n--- AI Correction Process ---\n";
            result << "\n--- Final Corrected Text ---\n";
            result << corrected << "\n";
        }

    } catch (const std::exception& e) {
        result << "\nError: " << e.what() << "\n";
    }

    std::string final_result = result.str();
    std::cout << final_result << std::endl;
    return final_result;
}
