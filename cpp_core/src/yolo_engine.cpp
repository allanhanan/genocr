#include "yolo_engine.hpp"
#include <algorithm>
#include <iostream>  // For debug prints

YoloEngine::YoloEngine(Ort::Env& env, Ort::SessionOptions& session_options, const std::string& model_path) {
    std::cout << "Loading YOLOv8 context detector from: " << model_path << std::endl;
    session_ = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    input_names_str_.push_back(session_->GetInputNameAllocated(0, allocator).get());
    output_names_str_.push_back(session_->GetOutputNameAllocated(0, allocator).get());
    input_names_.push_back(input_names_str_[0].c_str());
    output_names_.push_back(output_names_str_[0].c_str());

    class_names_ = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };
}

std::vector<std::string> YoloEngine::Detect(const cv::Mat& image) {
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(640, 640));
    cv::Mat blob = cv::dnn::blobFromImage(resized_image, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> yolo_input_dims = {1, 3, 640, 640};
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), blob.total(), yolo_input_dims.data(), yolo_input_dims.size()
    );

    auto outputs = session_->Run(
        Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1, output_names_.data(), 1
    );

    // Extract output data
    const float* output_data = outputs[0].template GetTensorData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();

    if (output_shape.size() != 3) {
        std::cerr << "Invalid output shape size: " << output_shape.size() << std::endl;
        return {};
    }

    int num_classes = 80;
    int num_detections = output_shape[2];  // 8400
    int num_params = output_shape[1];      // 84 (4 bbox + 80 classes)

    // Collect valid detections
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    
    float conf_threshold = 0.25f;  // Lower threshold for better detection
    float scale_x = static_cast<float>(image.cols) / 640.0f;
    float scale_y = static_cast<float>(image.rows) / 640.0f;

    // YOLOv8 output is transposed: [batch, 84, 8400]
    for (int i = 0; i < num_detections; ++i) {
        float cx = output_data[i];                    // center_x
        float cy = output_data[num_detections + i];   // center_y  
        float w = output_data[2 * num_detections + i]; // width
        float h = output_data[3 * num_detections + i]; // height
        
        // Find best class from remaining 80 class scores
        float max_class_score = 0.0f;
        int best_class_id = 0;
        
        for (int c = 0; c < num_classes; ++c) {
            float class_score = output_data[(4 + c) * num_detections + i];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                best_class_id = c;
            }
        }
        
        // Filter by confidence and size
        if (max_class_score > conf_threshold && w > 5 && h > 5) {
            // Convert to corner format and scale back to original image
            float x1 = (cx - w/2) * scale_x;
            float y1 = (cy - h/2) * scale_y;
            float width = w * scale_x;
            float height = h * scale_y;
            
            // Clamp to image boundaries
            x1 = std::max(0.0f, std::min(x1, static_cast<float>(image.cols - 1)));
            y1 = std::max(0.0f, std::min(y1, static_cast<float>(image.rows - 1)));
            width = std::min(width, static_cast<float>(image.cols) - x1);
            height = std::min(height, static_cast<float>(image.rows) - y1);
            
            if (width > 0 && height > 0) {
                boxes.emplace_back(static_cast<int>(x1), static_cast<int>(y1), 
                                 static_cast<int>(width), static_cast<int>(height));
                scores.push_back(max_class_score);
                class_ids.push_back(best_class_id);
            }
        }
    }

    // Apply Non-Maximum Suppression
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_threshold, 0.4f, indices);

    // Collect final detections with safety checks
    std::vector<std::string> context_clues;
    for (int idx : indices) {
        if (idx >= 0 && idx < static_cast<int>(class_ids.size())) {
            int cid = class_ids[idx];
            if (cid >= 0 && cid < static_cast<int>(class_names_.size())) {
                context_clues.push_back(class_names_[cid]);
            } else {
                std::cerr << "Warning: class id " << cid << " out of range." << std::endl;
            }
        } else {
            std::cerr << "Warning: NMS index " << idx << " out of range." << std::endl;
        }
    }

    // Remove duplicates
    std::sort(context_clues.begin(), context_clues.end());
    context_clues.erase(std::unique(context_clues.begin(), context_clues.end()), context_clues.end());
    
    return context_clues;
}
