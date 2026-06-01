/**
 * postprocess.hpp
 * Dynamic YOLO26 Post-Processing for Hailo-8L Outputs.
 * Supports arbitrary class counts and dynamic tensor shape mapping.
 */

#ifndef POSTPROCESS_HPP
#define POSTPROCESS_HPP

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <map>

// Struct for raw bounding box predictions
struct Detection {
    float x1, y1, x2, y2;
    float conf;
    int cls_id;
};

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * Identify output tensors by dynamically calculating their expected size 
 * based on the number of classes the model was trained on.
 */
inline bool map_output_tensors(
    const std::map<std::string, std::vector<float>>& output_buffers,
    std::vector<const float*>& cls_ptrs,
    std::vector<const float*>& reg_ptrs,
    int num_classes) 
{
    cls_ptrs.resize(3, nullptr);
    reg_ptrs.resize(3, nullptr);

    // Expected sizes for Regression (4 bounding box coordinates per anchor)
    size_t expected_reg_80 = 80 * 80 * 4; // 25,600
    size_t expected_reg_40 = 40 * 40 * 4; // 6,400
    size_t expected_reg_20 = 20 * 20 * 4; // 1,600

    // Expected sizes for Classification (num_classes logits per anchor)
    size_t expected_cls_80 = 80 * 80 * num_classes;
    size_t expected_cls_40 = 40 * 40 * num_classes;
    size_t expected_cls_20 = 20 * 20 * num_classes;

    for (const auto& pair : output_buffers) {
        size_t count = pair.second.size();
        
        if      (count == expected_cls_80) cls_ptrs[0] = pair.second.data();
        else if (count == expected_cls_40) cls_ptrs[1] = pair.second.data();
        else if (count == expected_cls_20) cls_ptrs[2] = pair.second.data();
        else if (count == expected_reg_80) reg_ptrs[0] = pair.second.data();
        else if (count == expected_reg_40) reg_ptrs[1] = pair.second.data();
        else if (count == expected_reg_20) reg_ptrs[2] = pair.second.data();
    }
    
    // Verify all 6 tensors were successfully found and mapped
    if (!cls_ptrs[0] || !cls_ptrs[1] || !cls_ptrs[2] || 
        !reg_ptrs[0] || !reg_ptrs[1] || !reg_ptrs[2]) {
        return false;
    }
    return true;
}

template <int... Is>
struct IntList {};

/**
 * Executes the NMS-free dual-head logic across the dynamically mapped pointers.
 */
template <int... Strides, int... Grids>
std::vector<Detection> run_postprocess(
    IntList<Strides...>,
    IntList<Grids...>,
    const std::vector<const float*>& cls_tensors,
    const std::vector<const float*>& reg_tensors,
    float conf_threshold,
    int num_classes
) {
    const int strides[] = {Strides...};
    constexpr int grids[] = {Grids...};
    constexpr size_t NUM_SCALES = sizeof...(Strides);
    
    static_assert(sizeof...(Strides) == sizeof...(Grids), "Strides and Grids lists must be the same length");
    
    if (cls_tensors.size() != NUM_SCALES || reg_tensors.size() != NUM_SCALES) {
        std::cerr << "Error: Expected " << NUM_SCALES << " class and regression tensors, got " 
                  << cls_tensors.size() << " and " << reg_tensors.size() << std::endl;
        return {};
    }

    std::vector<Detection> results;
    
    // Clamp threshold to avoid mathematical errors in logit calculation
    if (conf_threshold <= 0.0f) conf_threshold = 0.001f;
    if (conf_threshold >= 1.0f) conf_threshold = 0.999f;
    
    float logit_threshold = -std::log(1.0f / conf_threshold - 1.0f);

    for (size_t s = 0; s < NUM_SCALES; ++s) {

        int grid_dim = grids[s];
        int num_anchors = grid_dim * grid_dim;
        const float* cls_data = cls_tensors[s];
        const float* reg_data = reg_tensors[s];

        for (int i = 0; i < num_anchors; ++i) {
            float max_logit = -1000.0f; 
            int class_id = -1;
            int anchor_offset = i * num_classes; // Dynamic offset step
            
            for (int c = 0; c < num_classes; ++c) {
                float logit = cls_data[anchor_offset + c];
                if (logit > max_logit) {
                    max_logit = logit;
                    class_id = c;
                }
            }
            
            if (max_logit > logit_threshold) {
                float score = sigmoid(max_logit);
                
                int reg_offset = i * 4;
                float l = reg_data[reg_offset + 0];
                float t = reg_data[reg_offset + 1];
                float r = reg_data[reg_offset + 2];
                float b = reg_data[reg_offset + 3];
                
                int row = i / grid_dim;
                int col = i % grid_dim;
                
                float stride = (float)strides[s];
                
                // Decode Grid Bounding Boxes
                float x1 = (col + 0.5f - l) * stride;
                float y1 = (row + 0.5f - t) * stride;
                float x2 = (col + 0.5f + r) * stride;
                float y2 = (row + 0.5f + b) * stride;
                
                Detection det;
                det.x1 = x1;
                det.y1 = y1;
                det.x2 = x2; 
                det.y2 = y2; 
                det.conf = score;
                det.cls_id = class_id;
                
                results.push_back(det);
            }
        }
    }
    
    return results;
}

#endif // POSTPROCESS_HPP