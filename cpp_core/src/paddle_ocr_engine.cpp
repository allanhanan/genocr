#include "paddle_ocr_engine.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>

PaddleOcrEngine::PaddleOcrEngine(Ort::Env& env, Ort::SessionOptions& session_options, const std::string& models_dir) {
    Ort::AllocatorWithDefaultOptions allocator;

    std::cout << "Loading PaddleOCR detection model..." << std::endl;
    std::string det_path = models_dir + "/ocr_detection_multilingual/inference.onnx";
    det_session_ = std::make_unique<Ort::Session>(env, det_path.c_str(), session_options);

    std::cout << "Loading PaddleOCR recognition model..." << std::endl;
    std::string rec_path = models_dir + "/ocr_recognition_multilingual/inference.onnx";
    rec_session_ = std::make_unique<Ort::Session>(env, rec_path.c_str(), session_options);
    
    det_input_names_str_.push_back(det_session_->GetInputNameAllocated(0, allocator).get());
    det_output_names_str_.push_back(det_session_->GetOutputNameAllocated(0, allocator).get());
    det_input_names_.push_back(det_input_names_str_[0].c_str());
    det_output_names_.push_back(det_output_names_str_[0].c_str());

    rec_input_names_str_.push_back(rec_session_->GetInputNameAllocated(0, allocator).get());
    rec_output_names_str_.push_back(rec_session_->GetOutputNameAllocated(0, allocator).get());
    rec_input_names_.push_back(rec_input_names_str_[0].c_str());
    rec_output_names_.push_back(rec_output_names_str_[0].c_str());

    std::string dict_path = models_dir + "/ocr_recognition_multilingual/ppocrv5_dict.txt";
    LoadCharset(dict_path);
}

void PaddleOcrEngine::LoadCharset(const std::string& path) {
    std::ifstream file(path, std::ios::binary);  // Use binary to avoid newline translation
    if (!file.is_open()) {
        throw std::runtime_error("Could not open charset file: " + path);
    }

    charset_.clear();                     // Make sure we're starting fresh
    charset_.push_back("blank");          // CTC blank token

    std::string line;
    while (std::getline(file, line)) {
        // Remove trailing \r or \n (handles Windows and Unix)
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
            line.pop_back();
        }
        if (!line.empty()) {
            charset_.push_back(line);
        }
    }

    charset_.push_back("<EOS>");  // End-of-sequence marker
    std::cout << "Loaded charset with " << charset_.size() << " characters." << std::endl;
}


std::vector<PaddleUtils::OcrResult> PaddleOcrEngine::ExtractText(const cv::Mat& image) {
    std::vector<PaddleUtils::OcrResult> results;
    
    auto [resized, scale] = PaddleUtils::ResizeKeepRatio(image);
    cv::Mat blob = PaddleUtils::NormalizeImageNet(resized);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> det_input_dims = {1, 3, static_cast<long long>(resized.rows), static_cast<long long>(resized.cols)};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), blob.total(), det_input_dims.data(), det_input_dims.size()
    );
    
    auto det_outputs = det_session_->Run(
        Ort::RunOptions{nullptr}, det_input_names_.data(), &input_tensor, 1, det_output_names_.data(), 1
    );

    const float* prob_map = det_outputs[0].template GetTensorData<float>();
    std::vector<std::vector<cv::Point>> boxes = PaddleUtils::PostprocessDetection(prob_map, image.size(), resized.size(), scale);
    
    std::vector<std::string> full_charset_for_decode = {"blank"};
    full_charset_for_decode.insert(full_charset_for_decode.end(), charset_.begin(), charset_.end());
    full_charset_for_decode.push_back("<EOS>");

    for (const auto& box : boxes) {
        cv::Mat crop = PaddleUtils::WarpQuad(image, box);
        if (crop.empty()) continue;
        
        std::vector<cv::Mat> chunks = PaddleUtils::SplitWideCrop(crop);
        std::string full_text;
        std::vector<std::string> recognized_texts;
        
        for (const auto& chunk : chunks) {
            cv::Mat rec_blob = PaddleUtils::PreprocessRecognition(chunk);
            std::vector<int64_t> rec_input_dims = {1, 3, 48, 320};
            Ort::Value rec_input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, rec_blob.ptr<float>(), rec_blob.total(), rec_input_dims.data(), rec_input_dims.size()
            );

            auto rec_outputs = rec_session_->Run(
                Ort::RunOptions{nullptr}, rec_input_names_.data(), &rec_input_tensor, 1, rec_output_names_.data(), 1
            );
            
            const float* preds = rec_outputs[0].template GetTensorData<float>();
            auto preds_shape = rec_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            recognized_texts.push_back(PaddleUtils::DecodeRecognition(preds, preds_shape, full_charset_for_decode));
        }

        std::stringstream ss;
        for(size_t i = 0; i < recognized_texts.size(); ++i) {
            if (!recognized_texts[i].empty()) {
                if (ss.tellp() > 0) ss << " ";
                ss << recognized_texts[i];
            }
        }
        full_text = ss.str();
        results.push_back({box, full_text});
    }
    return results;
}