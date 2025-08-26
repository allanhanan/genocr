#include "corrector_engine.hpp"
#include <iostream>
#include <sstream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <regex>  // For post-processing

namespace {
// Proper limits for 128k context model
constexpr size_t HARD_CONTEXT_CAP = 128000;
constexpr size_t SAFETY_MARGIN = 1000; 
constexpr double DEFAULT_TEMPERATURE = 0.3;
constexpr double DEFAULT_TOP_P = 0.8;

std::filesystem::path make_temp_png_path() {
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto tmp_dir = std::filesystem::temp_directory_path();
    return tmp_dir / ("oga_img_" + std::to_string(now) + ".png");
}
}

CorrectorEngine::CorrectorEngine(const std::string& model_dir) {
    std::cout << "Loading Phi-3 Vision corrector model from: " << model_dir << std::endl;
    model_ = OgaModel::Create(model_dir.c_str());
    processor_ = OgaMultiModalProcessor::Create(*model_);
    tokenizer_stream_ = OgaTokenizerStream::Create(*processor_);
}

std::string CorrectorEngine::Correct(
    const cv::Mat& image,
    const std::vector<std::string>& context_clues,
    const std::string& raw_text)
{
    try {
        // Reasonable limit for 128k model
        if (raw_text.length() > 10000) {
            std::cerr << "[Corrector] Input text very long (" << raw_text.length() << " chars), proceeding with caution" << std::endl;
        }

        // 1) Build context string
        std::string context_str;
        context_str.reserve(512);
        for (const auto& clue : context_clues) {
            if (!clue.empty()) {
                context_str += "- " + clue + "\n";
                if (context_str.size() > 1000) break;
            }
        }

        // 2) Moderate image resize to reduce tokens
        cv::Mat resized_image;
        int max_dimension = 512;
        if (image.rows > max_dimension || image.cols > max_dimension) {
            double scale = static_cast<double>(max_dimension) / std::max(image.rows, image.cols);
            cv::resize(image, resized_image, cv::Size(0, 0), scale, scale, cv::INTER_LINEAR);
        } else {
            resized_image = image.clone();
        }

        const std::filesystem::path tmp_path = make_temp_png_path();
        if (!cv::imwrite(tmp_path.string(), resized_image)) {
            throw std::runtime_error("Failed to write temp image for OgaImages::Load");
        }

        std::string tmp_file_str = tmp_path.string();
        std::vector<const char*> image_paths = { tmp_file_str.c_str() };
        
        // 3) Load image(s)
        auto images = OgaImages::Load(image_paths);

        // 4) Language-agnostic prompt for multi-language support
        std::stringstream prompt_ss;
        prompt_ss
            << "<|user|>\n<|image_1|>\n"
            << "You are an expert in correcting OCR errors across multiple languages. Use the image and context to fix the raw text.\n"
            << "Rules:\n"
            << "- Preserve original structure with newlines.\n"
            << "- Fix casing, punctuation, spacing, accents, and common OCR misreads in any language.\n"
            << "- Reconstruct unclear parts plausibly from context and image, without adding new content.\n"
            << "- Output a single cohesive corrected passage — no explanations.\n"
            << "CONTEXT CLUES:\n" << context_str
            << "RAW TEXT:\n" << raw_text << "\n"
            << "<|end|>\n<|assistant|>\n";

        const std::string prompt_text = prompt_ss.str();

        // 5) Process prompt + images into model inputs
        std::unique_ptr<OgaNamedTensors> inputs;
        inputs = processor_->ProcessImages(prompt_text.c_str(), images.get());

        // 6) Clean up temp image
        std::filesystem::remove(tmp_path);

        // 7) Get prompt token count
        std::unique_ptr<OgaTensor> input_ids_tensor = inputs->Get("input_ids");
        if (!input_ids_tensor) {
            std::cerr << "[Corrector] Error: Failed to get input_ids tensor" << std::endl;
            return raw_text;
        }

        auto shape = input_ids_tensor->Shape();
        if (shape.size() < 2) {
            std::cerr << "[Corrector] Error: Unexpected input_ids tensor shape" << std::endl;
            return raw_text;
        }

        size_t prompt_tokens = static_cast<size_t>(shape[1]);
        
        // 8) Proper limits for 128k model
        if (prompt_tokens > HARD_CONTEXT_CAP - SAFETY_MARGIN) {
            std::cerr << "[Corrector] Prompt tokens (" << prompt_tokens << ") exceed limit, returning original text" << std::endl;
            return raw_text;
        }

        // Generous output tokens, scaled with input size
        size_t desired_output_tokens = std::min(static_cast<size_t>(raw_text.length() * 2), static_cast<size_t>(HARD_CONTEXT_CAP - prompt_tokens - SAFETY_MARGIN));

        size_t max_length = prompt_tokens + desired_output_tokens;

        std::cerr << "[Debug] Prompt tokens: " << prompt_tokens 
                  << ", Output: " << desired_output_tokens
                  << ", Max length: " << max_length << std::endl;

        // 9) Set generation parameters - neutral to avoid crash
        auto params = OgaGeneratorParams::Create(*model_);
        
        params->SetSearchOption("max_length", static_cast<int>(max_length));
        params->SetSearchOption("temperature", DEFAULT_TEMPERATURE);
        params->SetSearchOption("top_p", DEFAULT_TOP_P);
        params->SetSearchOption("repetition_penalty", 1.0);  // Neutral
        params->SetSearchOption("no_repeat_ngram_size", 0);  // Neutral
        params->SetSearchOption("top_k", 10);  // Add top-k limiting

        // 10) Create generator
        std::unique_ptr<OgaGenerator> generator = OgaGenerator::Create(*model_, *params);
        generator->SetInputs(*inputs);

        // 11) Generation loop
        int generation_steps = 0;
        const int MAX_GENERATION_STEPS =150;
        
        while (!generator->IsDone() && generation_steps < MAX_GENERATION_STEPS) {
            generator->GenerateNextToken();
            generation_steps++;
        }

        // 12) Decode output
        size_t sequence_count = generator->GetSequenceCount(0);
        if (sequence_count == 0) {
            return raw_text;
        }

        const int32_t* sequence_data = generator->GetSequenceData(0);
        if (!sequence_data) {
            return raw_text;
        }

        // Decode only new tokens
        std::string result = raw_text;  // Fallback
        if (sequence_count > prompt_tokens) {
            size_t new_tokens = sequence_count - prompt_tokens;
            OgaString oga_str = processor_->Decode(sequence_data + prompt_tokens, new_tokens);
            result = static_cast<const char*>(oga_str);
        }

        // 13) Post-processing for minor cleanup
        // Fix common accents and spacing
        result = std::regex_replace(result, std::regex("ē"), "é");
        result = std::regex_replace(result, std::regex("\\s+"), " ");  // Normalize spaces
        result = std::regex_replace(result, std::regex("^\\s+|\\s+$"), "");  // Trim
        result = std::regex_replace(result, std::regex("([.!?])\\s*([A-Z])"), "$1 $2");  // Sentence spacing

        return result;

    } catch (const std::exception& e) {
        std::cerr << "[Corrector] Fatal error in Correct(): " << e.what() << std::endl;
        return raw_text;
    }
}
