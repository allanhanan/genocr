#include "genocr.hpp"
#include "paddle_utils.hpp"  // Added to use WarpQuad for cropping
#include <string>

GenOCR::GenOCR(const std::string& models_directory, bool use_cuda) {
    // Shared ONNX Runtime environment for non-generative models
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "genocr-standard");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    if (use_cuda) {
        OrtCUDAProviderOptions cuda_options{};
        session_options.AppendExecutionProvider_CUDA(cuda_options);
    }

    yolo_engine_ = std::make_unique<YoloEngine>(env, session_options, models_directory + "/detector.onnx");
    paddle_engine_ = std::make_unique<PaddleOcrEngine>(env, session_options, models_directory);
    corrector_engine_ = std::make_unique<CorrectorEngine>(models_directory + "/corrector");
}

void GenOCR::Run(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image at: " << image_path << std::endl;
        return;
    }

    // --- Stage 1: Context Clues (YOLO) ---
    std::vector<std::string> context = yolo_engine_->Detect(image);

    // --- Stage 2: Raw Text (PaddleOCR) ---
    std::vector<PaddleUtils::OcrResult> ocr_results = paddle_engine_->ExtractText(image);

    // --- Stage 3: Correction (Phi-3) as single passage ---
    std::cout << "\n--- Final Corrected Results ---" << std::endl;
    
    // Build single raw text passage (concatenate with newlines for structure)
    std::string full_raw_text;
    for (const auto& ocr_result : ocr_results) {
        if (ocr_result.text.empty() || ocr_result.text.find_first_not_of(' ') == std::string::npos) continue;
        
        // Optional: Print individual raw for debugging
        std::cout << "RAW LINE: \"" << ocr_result.text << "\"" << std::endl;
        
        full_raw_text += ocr_result.text + "\n";
    }
    
    // Correct the entire passage in one call
    std::string corrected_passage = corrector_engine_->Correct(image, context, full_raw_text);
    
    // Print the single corrected passage
    std::cout << "\nCORRECTED PASSAGE:\n" << corrected_passage << std::endl;
}
