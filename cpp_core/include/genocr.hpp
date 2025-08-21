#pragma once
#include <string>
#include <vector>
#include <memory>
#include "yolo_engine.hpp"
#include "paddle_ocr_engine.hpp"
#include "corrector_engine.hpp"

class GenOCR {
public:
    GenOCR(const std::string& models_directory, bool use_cuda = false);
    void Run(const std::string& image_path);
private:
    std::unique_ptr<YoloEngine> yolo_engine_;
    std::unique_ptr<PaddleOcrEngine> paddle_engine_;
    std::unique_ptr<CorrectorEngine> corrector_engine_;
};