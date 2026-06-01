/**
 * detect_video.cpp
 * * Real-time video stream detection with Hailo-8L + Post-processing in C++.
 * Features dynamic class labeling, configurable model sizes, and standard argparse.
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
#include <cmath>
#include <algorithm>
#include <sstream>

#include "preprocess.hpp"
#include "postprocess.hpp"

using namespace hailort;

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

// Helper to print errors
void print_error(hailo_status status, const std::string& msg) {
    std::cerr << "Error: " << msg << " (Status: " << status << ")" << std::endl;
}

// ============================================================================
// Main Execution Pipeline
// ============================================================================
int main(int argc, char** argv) {
    InputParser input(argc, argv);

    if (input.cmdOptionExists("--help") || input.cmdOptionExists("-h")) {
        std::cout << "Usage: ./detect_video [OPTIONS]\n"
                  << "Options:\n"
                  << "  --hef <path>             Path to compiled HEF model (Default: ../models/yolo26n.hef)\n"
                  << "  --camera-id <id/url>     USB camera index or IP stream URL (Default: 0)\n"
                  << "  --labels <l1,l2>         Comma-separated list of class names (Default: bottle)\n"
                  << "  --model-size <int>       Model input square size (Default: 640)\n"
                  << "  --input-width <int>      Camera capture width (Default: 640)\n"
                  << "  --input-height <int>     Camera capture height (Default: 480)\n"
                  << "  --conf-threshold <float> Confidence threshold (Default: 0.25)\n";
        return 0;
    }

    // 1. Parse Arguments
    std::string hef_path = input.getCmdOption("--hef", "../models/yolo26n.hef");
    std::string camera_id = input.getCmdOption("--camera-id", "0");
    std::string labels_str = input.getCmdOption("--labels", "bottle");
    int model_size = std::stoi(input.getCmdOption("--model-size", "640"));
    int input_width = std::stoi(input.getCmdOption("--input-width", "640"));
    int input_height = std::stoi(input.getCmdOption("--input-height", "480"));
    float conf_threshold = std::stof(input.getCmdOption("--conf-threshold", "0.25"));

    std::vector<std::string> class_names = parse_labels(labels_str);

    std::cout << "[Initializing Hailo Engine for " << class_names.size() << " classes...]" << std::endl;
    
    // 2. Setup Hailo Hardware Device
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) { print_error(vdevice_exp.status(), "Failed to create VDevice"); return 1; }
    auto vdevice = std::move(vdevice_exp.value());

    auto hef_exp = Hef::create(hef_path);
    if (!hef_exp) { print_error(hef_exp.status(), "Failed to load HEF"); return 1; }
    auto hef = std::move(hef_exp.value());

    auto configure_params_exp = vdevice->create_configure_params(hef);
    auto configure_params = configure_params_exp.value();

    auto network_groups_exp = vdevice->configure(hef, configure_params);
    auto network_groups = std::move(network_groups_exp.value());
    auto network_group = network_groups[0];

    auto input_vstream_params = network_group->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE).value();
    auto output_vstream_params = network_group->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE).value();

    auto input_vstreams_exp = VStreamsBuilder::create_input_vstreams(*network_group, input_vstream_params);
    auto input_vstreams = std::move(input_vstreams_exp.value());

    auto output_vstreams_exp = VStreamsBuilder::create_output_vstreams(*network_group, output_vstream_params);
    auto output_vstreams = std::move(output_vstreams_exp.value());

    // Prepare output buffers map
    std::map<std::string, std::vector<float>> output_buffers;
    for (auto& ov : output_vstreams) {
        output_buffers[ov.name()] = std::vector<float>(ov.get_frame_size() / sizeof(float));
    }

    std::cout << "✓ Hailo Engine loaded successfully." << std::endl;

    // 3. Open Video Camera Stream
    cv::VideoCapture cap;
    if (camera_id.find_first_not_of("0123456789") == std::string::npos) {
        cap.open(std::stoi(camera_id)); // Open by device index
    } else {
        cap.open(camera_id); // Open by stream URL
    }

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera source: " << camera_id << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, input_width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, input_height);
    cap.set(cv::CAP_PROP_BUFFERSIZE, 1);

    std::cout << "[Starting video stream (" << input_width << "x" << input_height << "). Press 'q' to quit.]" << std::endl;

    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // A. Preprocess
        LetterboxInfo lb_info = letterbox_image(frame, model_size, model_size);
        cv::Mat resized_image = lb_info.image;
        cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

        auto t_infer_start = std::chrono::steady_clock::now();

        // B. Run Inference
        input_vstreams[0].write(MemoryView(resized_image.data, resized_image.total() * resized_image.elemSize()));
        
        std::vector<const float*> cls_ptrs(3, nullptr);
        std::vector<const float*> reg_ptrs(3, nullptr);
        int num_classes = class_names.size();

        for (auto& ov : output_vstreams) {
            ov.read(MemoryView(output_buffers[ov.name()].data(), output_buffers[ov.name()].size() * sizeof(float)));
            
            // Map tensors safely using Hardware Shape Metadata instead of raw buffer sizes
            auto shape = ov.get_info().shape;
            if (shape.features == num_classes) { // Classification Heads
                if (shape.height == 80) cls_ptrs[0] = output_buffers[ov.name()].data();
                else if (shape.height == 40) cls_ptrs[1] = output_buffers[ov.name()].data();
                else if (shape.height == 20) cls_ptrs[2] = output_buffers[ov.name()].data();
            } else if (shape.features == 4) { // Regression Heads (BBox coords)
                if (shape.height == 80) reg_ptrs[0] = output_buffers[ov.name()].data();
                else if (shape.height == 40) reg_ptrs[1] = output_buffers[ov.name()].data();
                else if (shape.height == 20) reg_ptrs[2] = output_buffers[ov.name()].data();
            }
        }

        auto t_infer_end = std::chrono::steady_clock::now();
        double infer_time_ms = std::chrono::duration<double, std::milli>(t_infer_end - t_infer_start).count();

        // Safety verification
        if (!cls_ptrs[0] || !cls_ptrs[1] || !cls_ptrs[2] || !reg_ptrs[0] || !reg_ptrs[1] || !reg_ptrs[2]) {
            std::cerr << "Error: Could not map HEF output tensors based on shape metadata!" << std::endl;
            break;
        }

        // C. C++ Post-Processing
        std::vector<Detection> results = run_postprocess(
            IntList<8, 16, 32>{}, IntList<80, 40, 20>{},
            cls_ptrs, reg_ptrs, conf_threshold, num_classes
        );

        // D. Draw Detections
        for (const auto& det : results) {
            // Reverse the letterbox padding to align with original camera frame
            float x1_f = (det.x1 - lb_info.pad_w) / lb_info.scale;
            float y1_f = (det.y1 - lb_info.pad_h) / lb_info.scale;
            float x2_f = (det.x2 - lb_info.pad_w) / lb_info.scale;
            float y2_f = (det.y2 - lb_info.pad_h) / lb_info.scale;
            
            // Safety Clipping
            int x1 = std::max(0, std::min((int)x1_f, frame.cols));
            int y1 = std::max(0, std::min((int)y1_f, frame.rows));
            int x2 = std::max(0, std::min((int)x2_f, frame.cols));
            int y2 = std::max(0, std::min((int)y2_f, frame.rows));
            
            // Draw bounding box
            cv::Rect rect(cv::Point(x1, y1), cv::Point(x2, y2));
            cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
            
            // Dynamically lookup label
            std::string class_name_str = (det.cls_id < class_names.size()) ? class_names[det.cls_id] : "Class_" + std::to_string(det.cls_id);
            char label[64];
            snprintf(label, sizeof(label), "%s (%.2f)", class_name_str.c_str(), det.conf);
            
            int font_face = cv::FONT_HERSHEY_TRIPLEX;
            double font_scale = 0.5;
            int thickness = 1;

            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);

            int text_y = rect.y - 5; 
            if (text_y - text_size.height < 0) {
                text_y = rect.y + text_size.height + 5; 
            }
            int text_x = std::max(0, rect.x);

            cv::Rect text_bg(text_x, text_y - text_size.height, text_size.width, text_size.height + baseline);
            cv::rectangle(frame, text_bg, cv::Scalar(255, 0, 0), cv::FILLED);
            cv::putText(frame, label, cv::Point(text_x, text_y), font_face, font_scale, cv::Scalar(0, 255, 0), thickness);
        }

        // E. Calculate and Draw FPS
        frame_count++;
        auto current_time = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(current_time - start_time).count();
        double fps = frame_count / elapsed_sec;

        cv::putText(frame, "FPS: " + std::to_string(fps).substr(0, 4), cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        cv::putText(frame, "Infer: " + std::to_string(infer_time_ms).substr(0, 4) + "ms", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 2);
        
        cv::imshow("Hailo C++ Real-Time Detector", frame);
        
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