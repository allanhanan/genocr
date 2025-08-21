#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

// This namespace encapsulates stateless helper functions for PaddleOCR.
namespace PaddleUtils {
    struct OcrResult {
        std::vector<cv::Point> box;
        std::string text;
    };

    std::pair<cv::Mat, float> ResizeKeepRatio(const cv::Mat& img, int max_side = 960);
    cv::Mat NormalizeImageNet(const cv::Mat& img);
    std::vector<std::vector<cv::Point>> PostprocessDetection(const float* prob_map, const cv::Size& original_shape, const cv::Size& resized_shape, float scale);
    cv::Mat WarpQuad(const cv::Mat& bgr_image, const std::vector<cv::Point>& quad);
    std::vector<cv::Mat> SplitWideCrop(const cv::Mat& crop, int chunk_width = 320, int overlap = 64);
    cv::Mat PreprocessRecognition(const cv::Mat& img);
    std::string DecodeRecognition(const float* preds, const std::vector<long>& preds_shape, const std::vector<std::string>& charset);
}