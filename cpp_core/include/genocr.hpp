#pragma once

#include <memory>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "yolo_engine.hpp"
#include "paddle_ocr_engine.hpp" 
#include "corrector_engine.hpp"

class GenOCR {
public:
    GenOCR(const std::string& models_directory, bool use_cuda = false);
    std::string Run(const std::string& image_path);

private:
    // Engine instances
    std::unique_ptr<YoloEngine> yolo_engine_;
    std::unique_ptr<PaddleOcrEngine> paddle_engine_;
    std::unique_ptr<CorrectorEngine> corrector_engine_;
    
    // Single shared environment 
    std::shared_ptr<Ort::Env> env_;
};
