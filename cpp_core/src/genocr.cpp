#include "genocr.hpp"
#include "paddle_utils.hpp"
#include <iostream>
#include <sstream>
#include <filesystem>
#include <future>
#include <thread>  // Added for std::thread::hardware_concurrency()

GenOCR::GenOCR(const std::string& models_directory, bool use_cuda) {
    // Shared ONNX Runtime environment for non-generative models
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "genocr-standard");
    Ort::SessionOptions session_options;
    
    // Smart threading: avoid thread oversubscription when running YOLO + OCR in parallel
    int num_cores = std::thread::hardware_concurrency();
    if (num_cores == 0) num_cores = 4; // Fallback if detection fails
    
    // Limit each model to use a reasonable portion of cores
    // Leave some cores free for system and other processes
    int threads_per_model = std::max(2, num_cores / 3);  // Use 1/3 of cores per model
    int inter_op_threads = std::max(1, num_cores / 6);   // Keep inter-op low to reduce overhead
    
    std::cout << "Threading config: " << num_cores << " cores detected, using " 
              << threads_per_model << " intra-op threads per model" << std::endl;
    
    session_options.SetIntraOpNumThreads(threads_per_model);  // Threads within each operation
    session_options.SetInterOpNumThreads(inter_op_threads);  // Threads for parallel operations
    
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options{};
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    yolo_engine_ = std::make_unique<YoloEngine>(env, session_options, models_directory + "/detector.onnx");
    paddle_engine_ = std::make_unique<PaddleOcrEngine>(env, session_options, models_directory);
    corrector_engine_ = std::make_unique<CorrectorEngine>(models_directory + "/corrector");
}

std::string GenOCR::Run(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::string error_msg = "Failed to load image at: " + image_path;
        std::cerr << error_msg << std::endl;
        return "Error: " + error_msg;
    }

    // Collect output in stringstream for return
    std::ostringstream result_stream;
    
    // Add header with filename
    std::string filename = std::filesystem::path(image_path).filename().string();
    result_stream << "=== OCR Results for " << filename << " ===\n";

    try {
        // --- Parallel Processing: Run YOLO and OCR simultaneously ---
        auto context_future = std::async(std::launch::async, [this, &image]() {
            return yolo_engine_->Detect(image);
        });
        
        auto ocr_future = std::async(std::launch::async, [this, &image]() {
            return paddle_engine_->ExtractText(image);
        });

        // Wait for both parallel tasks to complete
        std::vector<std::string> context = context_future.get();
        std::vector<PaddleUtils::OcrResult> ocr_results = ocr_future.get();

        // Display detected context objects
        if (!context.empty()) {
            result_stream << "\nDetected Context Objects:\n";
            for (const auto& obj : context) {
                result_stream << "- " << obj << "\n";
            }
        }

        // --- Stage 3: Correction (Phi-3) as single passage ---
        result_stream << "\n--- Raw OCR Output ---\n";
        
        // Build single raw text passage (concatenate with newlines for structure)
        std::string full_raw_text;
        for (const auto& ocr_result : ocr_results) {
            if (ocr_result.text.empty() || ocr_result.text.find_first_not_of(' ') == std::string::npos) continue;
            
            // Add to result stream for debugging
            result_stream << "RAW LINE: \"" << ocr_result.text << "\"\n";
            full_raw_text += ocr_result.text + "\n";
        }

        if (full_raw_text.empty()) {
            result_stream << "No text detected in image.\n";
            std::string final_result = result_stream.str();
            std::cout << final_result << std::endl;
            return final_result;
        }

        // Correct the entire passage in one call
        result_stream << "\n--- AI Correction Process ---\n";
        std::string corrected_passage = corrector_engine_->Correct(image, context, full_raw_text);

        result_stream << "\n--- Final Corrected Text ---\n";
        result_stream << corrected_passage << "\n";

    } catch (const std::exception& e) {
        std::string error_msg = "Error processing image: " + std::string(e.what());
        result_stream << "\n" << error_msg << "\n";
        std::cerr << error_msg << std::endl;
    }

    // Get final result
    std::string final_result = result_stream.str();
    
    // Print to console for immediate feedback
    std::cout << final_result << std::endl;
    
    // Return for CLI/batch processing
    return final_result;
}
