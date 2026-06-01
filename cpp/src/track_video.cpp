/**
 * track_video.cpp
 * Real-time object tracking with Hailo-8L + BoTSORT in C++.
 * Features dynamic class labeling, configurable model sizes, EMA smoothing,
 * pad inflation, and standard argparse.
 */

#include "hailo/hailort.hpp"
#include "hailo/vdevice.hpp"
#include "hailo/infer_model.hpp"
#include "hailo/vstream.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <map>
#include <set>
#include <cmath>
#include <algorithm>
#include <sstream>

// Unified Pre/Post-Processing
#include "preprocess.hpp"
#include "postprocess.hpp"

// External System Tracker Library
#include <roboflow_trackers/botsort_tracker.hpp>
#include <roboflow_trackers/detections.hpp>

using namespace hailort;
using namespace trackers::core::botsort;
using namespace trackers::utils;

// ============================================================================
// Command Line Argument Parser Helper
// ============================================================================
class InputParser {
public:
    InputParser(int &argc, char **argv) {
        for (int i = 1; i < argc; ++i)
            this->tokens.push_back(std::string(argv[i]));
    }
    const std::string& getCmdOption(const std::string &option, const std::string &default_val = "") const {
        std::vector<std::string>::const_iterator itr;
        itr = std::find(this->tokens.begin(), this->tokens.end(), option);
        if (itr != this->tokens.end() && ++itr != this->tokens.end()) {
            return *itr;
        }
        static const std::string empty_string("");
        return default_val.empty() ? empty_string : default_val;
    }
    bool cmdOptionExists(const std::string &option) const {
        return std::find(this->tokens.begin(), this->tokens.end(), option) != this->tokens.end();
    }
private:
    std::vector<std::string> tokens;
};

// Helper to parse comma-separated labels
std::vector<std::string> parse_labels(const std::string& labels_str) {
    std::vector<std::string> labels;
    std::stringstream ss(labels_str);
    std::string item;
    while (std::getline(ss, item, ',')) {
        labels.push_back(item);
    }
    return labels;
}

void print_error(hailo_status status, const std::string& msg) {
    std::cerr << "Error: " << msg << " (Status: " << status << ")" << std::endl;
}

// ============================================================================
// Main Execution Pipeline
// ============================================================================
int main(int argc, char** argv) {
    InputParser input(argc, argv);

    if (input.cmdOptionExists("--help") || input.cmdOptionExists("-h")) {
        std::cout << "Usage: ./track_video [OPTIONS]\n"
                  << "Options:\n"
                  << "  --hef <path>                 Path to compiled HEF model (Default: ../models/yolo26n.hef)\n"
                  << "  --camera-id <id/url>         USB camera index or IP stream URL (Default: 0)\n"
                  << "  --labels <l1,l2>             Comma-separated list of class names (Default: bottle)\n"
                  << "  --model-size <int>           Model input square size (Default: 640)\n"
                  << "  --input-width <int>          Camera capture width (Default: 640)\n"
                  << "  --input-height <int>         Camera capture height (Default: 480)\n"
                  << "  --conf-threshold <float>     Detection threshold (Default: 0.25)\n"
                  << "\nBoTSORT Tracker Options:\n"
                  << "  --track-buffer <int>         Frames to keep an inactive track (Default: 120)\n"
                  << "  --min-frames <int>           Min frames to activate an object path (Default: 1)\n"
                  << "  --high-conf-threshold <flt>  Threshold for first association pass (Default: 0.3)\n"
                  << "  --track-activation <flt>     Threshold to spin up brand new track (Default: 0.85)\n"
                  << "  --iou-first-assoc <flt>      IoU ceiling for primary assignment (Default: 0.1)\n"
                  << "  --iou-second-assoc <flt>     IoU ceiling for secondary assignment (Default: 0.4)\n"
                  << "\nSmoothing & Diagnostics:\n"
                  << "  --pad-inflation <int>        Pixel margin temporarily added to bounding boxes for association stability (Default: 40)\n"
                  << "  --alpha <float>              EMA smoothing filter [0.0=Static, 1.0=Raw] (Default: 0.7)\n"
                  << "  --debug                      Enable advanced tracker ID switch diagnostics\n";
        return 0;
    }

    // 1. Parse IO Arguments
    std::string hef_path = input.getCmdOption("--hef", "../models/yolo26n.hef");
    std::string camera_id = input.getCmdOption("--camera-id", "0");
    std::string labels_str = input.getCmdOption("--labels", "bottle");
    int model_size = std::stoi(input.getCmdOption("--model-size", "640"));
    int input_width = std::stoi(input.getCmdOption("--input-width", "640"));
    int input_height = std::stoi(input.getCmdOption("--input-height", "480"));
    float conf_threshold = std::stof(input.getCmdOption("--conf-threshold", "0.25"));
    std::vector<std::string> class_names = parse_labels(labels_str);

    // 2. Parse Tracker Hyperparameters
    int track_buffer = std::stoi(input.getCmdOption("--track-buffer", "120"));
    int min_frames = std::stoi(input.getCmdOption("--min-frames", "1"));
    float high_conf_det_threshold = std::stof(input.getCmdOption("--high-conf-threshold", "0.3"));
    float track_activation_threshold = std::stof(input.getCmdOption("--track-activation", "0.85"));
    float iou_first_assoc = std::stof(input.getCmdOption("--iou-first-assoc", "0.1"));
    float iou_second_assoc = std::stof(input.getCmdOption("--iou-second-assoc", "0.4"));
   
    // 3. Parse Smoothing & Hacks
    int pad_inflation = std::stoi(input.getCmdOption("--pad-inflation", "40"));
    float alpha = std::stof(input.getCmdOption("--alpha", "0.7"));
    bool debug_mode = input.cmdOptionExists("--debug");

    std::cout << "[Initializing Hailo Engine for " << class_names.size() << " classes...]" << std::endl;
   
    // Hardware Setup
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) { print_error(vdevice_exp.status(), "Failed to create VDevice"); return 1; }
    auto vdevice = std::move(vdevice_exp.value());

    auto hef_exp = Hef::create(hef_path);
    if (!hef_exp) { print_error(hef_exp.status(), "Failed to load HEF"); return 1; }
    auto hef = std::move(hef_exp.value());

    auto configure_params_exp = vdevice->create_configure_params(hef);
    auto network_groups = std::move(vdevice->configure(hef, configure_params_exp.value()).value());
    auto network_group = network_groups[0];

    auto input_vstream_params = network_group->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE).value();
    auto output_vstream_params = network_group->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE).value();

    auto input_vstreams_exp = VStreamsBuilder::create_input_vstreams(*network_group, input_vstream_params);
    auto input_vstreams = std::move(input_vstreams_exp.value());

    auto output_vstreams_exp = VStreamsBuilder::create_output_vstreams(*network_group, output_vstream_params);
    auto output_vstreams = std::move(output_vstreams_exp.value());

    std::map<std::string, std::vector<float>> output_buffers;
    for (auto& ov : output_vstreams) {
        output_buffers[ov.name()] = std::vector<float>(ov.get_frame_size() / sizeof(float));
    }

    std::cout << "✓ Hailo Engine loaded successfully." << std::endl;

    // 4. Initialize External Tracker (FIXED parameter order matching botsort_tracker.hpp)
    BoTSORTTracker tracker(
        track_buffer,                   // lost_track_buffer
        30.0f,                          // frame_rate (default)
        track_activation_threshold,     // track_activation_threshold
        min_frames,                     // minimum_consecutive_frames
        iou_first_assoc,                // minimum_iou_threshold_first_assoc
        iou_second_assoc,               // minimum_iou_threshold_second_assoc
        0.3f,                           // minimum_iou_threshold_unconfirmed_assoc
        high_conf_det_threshold         // high_conf_det_threshold
    );

    // 5. Open Camera
    cv::VideoCapture cap;
    if (camera_id.find_first_not_of("0123456789") == std::string::npos) {
        cap.open(std::stoi(camera_id));
    } else {
        cap.open(camera_id);
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera source: " << camera_id << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, input_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, input_height);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    std::cout << "[Starting tracking stream (" << input_width << "x" << input_height << "). Press 'q' to quit.]" << std::endl;

    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    // Data structures for EMA smoothing and ID swap diagnostics
    std::map<int, cv::Rect2f> track_history;
    std::map<int, std::tuple<float, float, float>> previous_frame_data;
    int last_known_id = -1;
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        LetterboxInfo lb_info = letterbox_image(frame, model_size, model_size);
        cv::Mat resized_image = lb_info.image;
        cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

        // A. Inference
        auto t_infer_start = std::chrono::steady_clock::now();
        input_vstreams[0].write(MemoryView(resized_image.data, resized_image.total() * resized_image.elemSize()));
        
        std::vector<const float*> cls_ptrs(3, nullptr);
        std::vector<const float*> reg_ptrs(3, nullptr);
        int num_classes = class_names.size(); 

        for (auto& ov : output_vstreams) {
            ov.read(MemoryView(output_buffers[ov.name()].data(), output_buffers[ov.name()].size() * sizeof(float)));
            
            auto shape = ov.get_info().shape;
            if (shape.features == num_classes) {
                if (shape.height == 80) cls_ptrs[0] = output_buffers[ov.name()].data();
                else if (shape.height == 40) cls_ptrs[1] = output_buffers[ov.name()].data();
                else if (shape.height == 20) cls_ptrs[2] = output_buffers[ov.name()].data();
            } else if (shape.features == 4) {
                if (shape.height == 80) reg_ptrs[0] = output_buffers[ov.name()].data();
                else if (shape.height == 40) reg_ptrs[1] = output_buffers[ov.name()].data();
                else if (shape.height == 20) reg_ptrs[2] = output_buffers[ov.name()].data();
            }
        }
        
        auto t_infer_end = std::chrono::steady_clock::now();
        double infer_time_ms = std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();

        // B. Post-Processing
        auto results = run_postprocess(
            IntList<8, 16, 32>{}, IntList<80, 40, 20>{},
            cls_ptrs, reg_ptrs, conf_threshold, num_classes
        );
        
        // C. Build Tracker Inputs (with Pad Inflation)
        std::vector<trackers::utils::Detection> tracker_inputs;
        for (const auto& det : results) {
            float x1_f = (det.x1 - lb_info.pad_w) / lb_info.scale;
            float y1_f = (det.y1 - lb_info.pad_h) / lb_info.scale;
            float x2_f = (det.x2 - lb_info.pad_w) / lb_info.scale;
            float y2_f = (det.y2 - lb_info.pad_h) / lb_info.scale;

            x1_f -= pad_inflation;
            y1_f -= pad_inflation;
            x2_f += pad_inflation;
            y2_f += pad_inflation;

            // FIXED: Using pure XYXY format, mapping `det.conf` to `confidence` (float), and `det.cls_id` to `class_id` (int)
            tracker_inputs.push_back(trackers::utils::Detection{
                cv::Vec4f(x1_f, y1_f, x2_f, y2_f),
                det.conf,  
                det.cls_id  
            });
        }

        // D. Update Tracker
        auto tracked_objects = tracker.update(tracker_inputs, frame);

        std::set<int> active_ids;

        // E. Coordinate Transforms, Smoothing & Visualizations
        for (auto& track : tracked_objects) {
           
            int tracker_id = track.tracker_id;
            active_ids.insert(tracker_id);
           
            // FIXED: Mapped explicitly to "confidence"
            float conf = track.confidence;
            int class_id = track.class_id;

            // FIXED: Bounding box from tracker is in XYXY format
            float x1 = track.bbox[0] + pad_inflation;
            float y1 = track.bbox[1] + pad_inflation;
            float x2 = track.bbox[2] - pad_inflation;
            float y2 = track.bbox[3] - pad_inflation;

            x1 = std::max(0.0f, std::min(x1, (float)frame.cols));
            y1 = std::max(0.0f, std::min(y1, (float)frame.rows));
            x2 = std::max(0.0f, std::min(x2, (float)frame.cols));
            y2 = std::max(0.0f, std::min(y2, (float)frame.rows));
           
            // Convert XYXY back to OpenCV Rect [x, y, w, h] for rendering
            cv::Rect2f current_raw_box(x1, y1, x2 - x1, y2 - y1);
           
            float w = x2 - x1;
            float h = y2 - y1;
            float cx = x1 + (w / 2.0f);
            float cy = y1 + (h / 2.0f);

            if (debug_mode && last_known_id != -1 && tracker_id != last_known_id) {
                std::cout << " [!!!] ID SWITCH DETECTED: " << last_known_id << " -> " << tracker_id << std::endl;
                if (previous_frame_data.count(last_known_id)) {
                    auto [prev_cx, prev_cy, prev_conf] = previous_frame_data[last_known_id];
                    float distance = std::sqrt(std::pow(cx - prev_cx, 2) + std::pow(cy - prev_cy, 2));
                    std::cout << "        Jump Distance: " << distance << " px" << std::endl;
                    std::cout << "        Prev Conf: " << prev_conf << " | Current Conf: " << conf << std::endl;
                }
            }

            previous_frame_data[tracker_id] = std::make_tuple(cx, cy, conf);
            last_known_id = tracker_id;

            cv::Rect2f smoothed_box = current_raw_box;
            if (track_history.count(tracker_id)) {
                cv::Rect2f prev_box = track_history[tracker_id];
                smoothed_box.x = (alpha * current_raw_box.x) + ((1.0f - alpha) * prev_box.x);
                smoothed_box.y = (alpha * current_raw_box.y) + ((1.0f - alpha) * prev_box.y);
                smoothed_box.width = (alpha * current_raw_box.width) + ((1.0f - alpha) * prev_box.width);
                smoothed_box.height = (alpha * current_raw_box.height) + ((1.0f - alpha) * prev_box.height);
            }
            track_history[tracker_id] = smoothed_box;

            cv::rectangle(frame, smoothed_box, cv::Scalar(255, 0, 0), 2);

            std::string class_name_str = (class_id < class_names.size()) ? class_names[class_id] : "Class_" + std::to_string(class_id);
            char label[64];
            snprintf(label, sizeof(label), "%s #%d (%.2f)", class_name_str.c_str(), tracker_id, conf);

            int font_face = cv::FONT_HERSHEY_TRIPLEX;
            double font_scale = 0.5;
            int thickness = 1;
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);

            int text_y = smoothed_box.y - 5;
            if (text_y - text_size.height < 0) text_y = smoothed_box.y + text_size.height + 5;
            int text_x = std::max(0.0f, smoothed_box.x);

            cv::Rect text_bg(text_x, text_y - text_size.height, text_size.width, text_size.height + baseline);
            cv::rectangle(frame, text_bg, cv::Scalar(255, 0, 0), cv::FILLED);
            cv::putText(frame, label, cv::Point(text_x, text_y), font_face, font_scale, cv::Scalar(0, 255, 0), thickness);
        }

        // F. Evict Stale Tracks
        for (auto it = track_history.begin(); it != track_history.end(); ) {
            if (active_ids.find(it->first) == active_ids.end()) it = track_history.erase(it);
            else ++it;
        }
        for (auto it = previous_frame_data.begin(); it != previous_frame_data.end(); ) {
            if (active_ids.find(it->first) == active_ids.end()) it = previous_frame_data.erase(it);
            else ++it;
        }

        // G. Render Telemetry
        frame_count++;
        auto current_time = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(current_time - start_time).count();
        double fps = frame_count / elapsed_sec;

        char fps_text[32], infer_text[32];
        snprintf(fps_text, sizeof(fps_text), "FPS: %.1f", fps);
        snprintf(infer_text, sizeof(infer_text), "Infer: %.1fms", infer_time_ms);

        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, infer_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        
        cv::imshow("Hailo-8L Tracking + BoTSORT", frame);
        // Press 'q' to quit
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            break;
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    std::cout << "[Video stream closed.]" << std::endl;
    return 0;
}