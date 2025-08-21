#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_genai.h>
#include "paddle_utils.hpp"

class CorrectorEngine {
public:
    CorrectorEngine(const std::string& model_dir);
    std::string Correct(const cv::Mat& image, const std::vector<std::string>& context_clues, const std::string& raw_text);
private:
    std::unique_ptr<OgaModel> model_;
    std::unique_ptr<OgaMultiModalProcessor> processor_;
    std::unique_ptr<OgaTokenizerStream> tokenizer_stream_;
};