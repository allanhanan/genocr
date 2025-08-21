#include "paddle_utils.hpp"
#include "clipper2/clipper.h"
#include <iostream>
#include <algorithm>

namespace PaddleUtils {

    std::pair<cv::Mat, float> ResizeKeepRatio(const cv::Mat& img, int max_side) {
        int h = img.rows;
        int w = img.cols;
        float scale = 1.0f;
        if (std::max(h, w) > max_side) {
            scale = static_cast<float>(max_side) / static_cast<float>(std::max(h, w));
        }
        
        int new_h = static_cast<int>(h * scale);
        int new_w = static_cast<int>(w * scale);
        
        if (new_h % 32 != 0) new_h = (new_h / 32 + 1) * 32;
        if (new_w % 32 != 0) new_w = (new_w / 32 + 1) * 32;

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(new_w, new_h));
        return {resized, scale};
    }

    cv::Mat NormalizeImageNet(const cv::Mat& img) {
        cv::Mat float_img;
        img.convertTo(float_img, CV_32F, 1.0 / 255.0);
        cv::Mat mean(float_img.size(), CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
        cv::Mat std(float_img.size(), CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
        cv::subtract(float_img, mean, float_img);
        cv::divide(float_img, std, float_img);
        return cv::dnn::blobFromImage(float_img);
    }
    
    double PathPerimeter(const Clipper2Lib::Path64& path) {
        double perimeter = 0.0;
        if (path.size() < 2) return 0.0;
        for (size_t i = 0; i < path.size(); ++i) {
            const auto& p1 = path[i];
            const auto& p2 = path[(i + 1) % path.size()];
            perimeter += std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
        }
        return perimeter;
    }

    std::vector<cv::Point2f> Unclip(const std::vector<cv::Point2f>& box, float ratio) {
        Clipper2Lib::Path64 path;
        // Convert floating point box to integer path for Clipper
        for (const auto& p : box) {
            path.push_back(Clipper2Lib::Point64(p.x, p.y));
        }

        double area = Clipper2Lib::Area(path);
        double length = PathPerimeter(path);
        if (length == 0) return box;

        double distance = area * ratio / length;
        Clipper2Lib::Paths64 solution;
        Clipper2Lib::ClipperOffset offset;
        offset.AddPath(path, Clipper2Lib::JoinType::Round, Clipper2Lib::EndType::Polygon);
        offset.Execute(distance, solution);

        std::vector<cv::Point2f> float_box;
        if (solution.empty() || solution[0].empty()) {
            return box; // Return original box if unclip fails
        }
        
        // Convert integer solution back to OpenCV's float point format
        std::vector<cv::Point> contour;
        for (const Clipper2Lib::Point64& p : solution[0]) {
            contour.push_back(cv::Point(static_cast<int>(p.x), static_cast<int>(p.y)));
        }
        cv::RotatedRect rect = cv::minAreaRect(contour);
        cv::Point2f points[4];
        rect.points(points);
        for(int i = 0; i < 4; ++i) float_box.push_back(points[i]);
        
        return float_box;
    }

    float BoxScore(const cv::Mat& prob_map, const std::vector<cv::Point2f>& box) {
        std::vector<cv::Point> int_box;
        for(const auto& p : box) int_box.push_back(cv::Point(p.x, p.y));
        
        cv::Mat mask = cv::Mat::zeros(prob_map.rows, prob_map.cols, CV_8U);
        cv::fillPoly(mask, {int_box}, 1);
        return cv::mean(prob_map, mask)[0];
    }
    
    std::vector<cv::Point2f> QuadFromContour(const std::vector<cv::Point>& contour) {
        cv::RotatedRect rect = cv::minAreaRect(contour);
        cv::Point2f points[4];
        rect.points(points);
        return {points[0], points[1], points[2], points[3]};
    }
    
    std::vector<std::vector<cv::Point>> PostprocessDetection(const float* prob_map, const cv::Size& original_shape, const cv::Size& resized_shape, float scale) {
        cv::Mat prob_mat(resized_shape.height, resized_shape.width, CV_32F, (void*)prob_map);
        cv::Mat bitmap = prob_mat > 0.3;

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        
        std::vector<std::vector<cv::Point>> boxes;
        for (const auto& contour : contours) {
            if (contour.size() < 4) continue;
            
            std::vector<cv::Point2f> box_f = QuadFromContour(contour);
            if (BoxScore(prob_mat, box_f) < 0.6) continue;
            
            std::vector<cv::Point2f> unclipped_box_f = Unclip(box_f, 1.5);
            
            std::vector<cv::Point> final_box;
            for(auto& p : unclipped_box_f) {
                p.x /= scale;
                p.y /= scale;
                p.x = std::min(std::max(0.0f, p.x), static_cast<float>(original_shape.width - 1));
                p.y = std::min(std::max(0.0f, p.y), static_cast<float>(original_shape.height - 1));
                final_box.push_back(cv::Point(p.x, p.y));
            }
            
            if (cv::contourArea(final_box) < 80) continue;
            
            boxes.push_back(final_box);
        }
        std::sort(boxes.begin(), boxes.end(), [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
            return a[0].y < b[0].y;
        });
        return boxes;
    }
    
    cv::Mat WarpQuad(const cv::Mat& bgr_image, const std::vector<cv::Point>& quad) {
        if (quad.size() != 4) return cv::Mat();
        std::vector<cv::Point2f> quad_f;
        for(const auto& p : quad) quad_f.push_back(cv::Point2f(p.x, p.y));

        float w = std::max(cv::norm(quad_f[0] - quad_f[1]), cv::norm(quad_f[2] - quad_f[3]));
        float h = std::max(cv::norm(quad_f[0] - quad_f[3]), cv::norm(quad_f[1] - quad_f[2]));
        if (w <= 0 || h <= 0) return cv::Mat();

        std::vector<cv::Point2f> dst_pts = {{0, 0}, {w, 0}, {w, h}, {0, h}};
        cv::Mat M = cv::getPerspectiveTransform(quad_f, dst_pts);
        
        cv::Mat crop;
        cv::warpPerspective(bgr_image, crop, M, cv::Size(static_cast<int>(w), static_cast<int>(h)), cv::INTER_CUBIC, cv::BORDER_REPLICATE);

        if (crop.rows > 0 && crop.cols > 0 && (static_cast<float>(crop.rows) / crop.cols >= 1.5)) {
            cv::rotate(crop, crop, cv::ROTATE_90_COUNTERCLOCKWISE);
        }
        return crop;
    }
    
    std::vector<cv::Mat> SplitWideCrop(const cv::Mat& crop, int chunk_width, int overlap) {
        if (crop.cols <= chunk_width) return {crop};
        std::vector<cv::Mat> chunks;
        for (int start = 0; start < crop.cols; start += (chunk_width - overlap)) {
            int end = std::min(start + chunk_width, crop.cols);
            chunks.push_back(crop(cv::Rect(start, 0, end - start, crop.rows)).clone());
            if (end == crop.cols) break;
        }
        return chunks;
    }
    
cv::Mat PreprocessRecognition(const cv::Mat& img) {
    if (img.empty()) return cv::Mat();

    // Use your constants if available (REC_H/REC_W). Replace these two lines if needed.
    const int rec_h = 48;
    const int rec_w = 320;

    // Ensure 3 channels (old path assumed 3-channel input)
    cv::Mat src = img;
    if (src.channels() == 1) {
        cv::cvtColor(src, src, cv::COLOR_GRAY2BGR);
    }

    // Maintain aspect ratio, compute target width
    const float ratio = static_cast<float>(src.cols) / static_cast<float>(src.rows);
    int resized_w = static_cast<int>(std::ceil(rec_h * ratio));
    if (resized_w <= 0) return cv::Mat();

    // Resize exactly like Python (INTER_LINEAR)
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(resized_w, rec_h), 0, 0, cv::INTER_LINEAR);

    // Convert to float [0,1]
    cv::Mat norm_img;
    resized.convertTo(norm_img, CV_32FC3, 1.0 / 255.0);

    // Zero-pad to (rec_h, rec_w, 3)
    cv::Mat padding_img = cv::Mat::zeros(rec_h, rec_w, CV_32FC3);
    int width_to_copy = std::min(resized_w, rec_w);

    if (width_to_copy > 0 &&
        width_to_copy <= norm_img.cols &&
        rec_h <= norm_img.rows) {
        norm_img(cv::Rect(0, 0, width_to_copy, rec_h))
            .copyTo(padding_img(cv::Rect(0, 0, width_to_copy, rec_h)));
    }

    // Create blob in NCHW without channel swap (matches old CHW ordering)
    // Note: swapRB=false (do NOT convert BGR->RGB), scalefactor=1.0 (already normalized)
    return cv::dnn::blobFromImage(
        padding_img,      // image
        1.0,              // scalefactor
        cv::Size(),       // size (use actual image size)
        cv::Scalar(),     // mean
        false,            // swapRB (must be false to match old)
        false,            // crop
        CV_32F            // ddepth
    );
}

    
    std::string DecodeRecognition(const float* preds, const std::vector<long>& preds_shape, const std::vector<std::string>& charset_with_blank) {
        std::string text = "";
        long last_idx = 0;
        long num_tokens = preds_shape[1];
        long num_chars = preds_shape[2];

        for (long t = 0; t < num_tokens; ++t) {
            long max_idx = 0;
            float max_prob = -1.0f;
            for (long c = 0; c < num_chars; ++c) {
                if (preds[t * num_chars + c] > max_prob) {
                    max_prob = preds[t * num_chars + c];
                    max_idx = c;
                }
            }
            if (max_idx > 0 && max_idx != last_idx) {
                long shifted_idx = max_idx+1;
                if (shifted_idx < charset_with_blank.size()) {
                    if (charset_with_blank[shifted_idx] == "<EOS>") {
                        text += " ";
                    } else {
                        text += charset_with_blank[shifted_idx];
                    }
                }
            }
            last_idx = max_idx;
        }
        return text;
    }
}