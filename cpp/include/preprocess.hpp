/**
 * preprocess.hpp
 * Image normalization and letterboxing utilities for YOLO architectures.
 */

#pragma once

#include <opencv2/opencv.hpp>
#include <tuple>
#include <algorithm>

// Struct to store padding margins for scaling boxes back to original view
struct LetterboxInfo {
    cv::Mat image;
    float scale;
    int pad_w;
    int pad_h;
};

/**
 * Resizes an image to target dimensions while maintaining its aspect ratio.
 * Pads the remaining space with a solid color (default gray).
 */
inline LetterboxInfo letterbox_image(const cv::Mat& img, int target_w, int target_h, const cv::Scalar& color = cv::Scalar(114, 114, 114)) {
    int h = img.rows;
    int w = img.cols;
    
    float scale = std::min((float)target_w / w, (float)target_h / h);
    int new_w = (int)(w * scale);
    int new_h = (int)(h * scale);
    
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));
    
    int pad_w = (target_w - new_w) / 2;
    int pad_h = (target_h - new_h) / 2;
    
    cv::Mat padded(target_h, target_w, CV_8UC3, color);
    
    // Copy resized image into the exact center of the padded canvas
    resized.copyTo(padded(cv::Rect(pad_w, pad_h, new_w, new_h)));
    
    return {padded, scale, pad_w, pad_h};
}