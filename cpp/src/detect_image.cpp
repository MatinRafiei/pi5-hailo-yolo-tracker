/**
 * detect_image.cpp
 * * Single image detection with Hailo-8L + Post-processing in C++.
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
#include <memory>
#include <map>
#include <algorithm>
#include <sstream>

#include "postprocess.hpp"
#include "preprocess.hpp"

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
    std::string getPositional(int index) const {
        // Find the first token that doesn't start with '--' and isn't a value for a flag
        int pos_count = 0;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (tokens[i].rfind("--", 0) == 0) {
                i++; // skip the flag's value
                continue;
            }
            if (pos_count == index) return tokens[i];
            pos_count++;
        }
        return "";
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

    // Print Help
    if (input.cmdOptionExists("--help") || input.cmdOptionExists("-h") || argc < 2) {
        std::cout << "Usage: ./detect_image <image_path> [OPTIONS]\n"
                  << "Options:\n"
                  << "  --hef <path>             Path to compiled HEF model (Required)\n"
                  << "  --output <path>          Output image path (Default: output_detected.jpg)\n"
                  << "  --labels <l1,l2>         Comma-separated list of class names (Default: bottle)\n"
                  << "  --model-size <int>       Model input square size (Default: 640)\n"
                  << "  --conf-threshold <float> Confidence threshold (Default: 0.25)\n";
        return 1;
    }

    // 1. Parse Arguments
    std::string image_path = input.getPositional(0);
    std::string hef_path = input.getCmdOption("--hef", "");
    std::string output_path = input.getCmdOption("--output", "output_detected.jpg");
    std::string labels_str = input.getCmdOption("--labels", "bottle");
    int model_size = std::stoi(input.getCmdOption("--model-size", "640"));
    float conf_threshold = std::stof(input.getCmdOption("--conf-threshold", "0.25"));
   
    std::vector<std::string> class_names = parse_labels(labels_str);

    if (hef_path.empty()) {
        std::cerr << "Error: --hef is required." << std::endl;
        return 1;
    }

    std::cout << "[Initializing Hailo Engine for " << class_names.size() << " classes...]" << std::endl;
    std::cout << "[Loading image: " << image_path << "]" << std::endl;

    // 2. Preprocess Image (Letterbox)
    cv::Mat orig_image = cv::imread(image_path);
    if (orig_image.empty()) {
        std::cerr << "Error: Could not read image: " << image_path << std::endl;
        return 1;
    }

    std::cout << "✓ Original image size: " << orig_image.cols << "x" << orig_image.rows << std::endl;

    LetterboxInfo lb_info = letterbox_image(orig_image, model_size, model_size);
    cv::Mat resized_image = lb_info.image;
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);

    std::cout << "✓ Preprocessed to: (" << resized_image.rows << ", " << resized_image.cols << ", " << resized_image.channels() << ")" << std::endl;

    // 3. Setup Hailo Hardware Device
    auto vdevice_exp = VDevice::create();
    if (!vdevice_exp) { print_error(vdevice_exp.status(), "Failed to create VDevice"); return 1; }
    auto vdevice = std::move(vdevice_exp.value());

    auto hef_exp = Hef::create(hef_path);
    if (!hef_exp) { print_error(hef_exp.status(), "Failed to load HEF"); return 1; }
    auto hef = std::move(hef_exp.value());

    auto configure_params_exp = vdevice->create_configure_params(hef);
    if (!configure_params_exp) { print_error(configure_params_exp.status(), "Failed to create configure params"); return 1; }
    auto configure_params = configure_params_exp.value();

    auto network_groups_exp = vdevice->configure(hef, configure_params);
    if (!network_groups_exp) { print_error(network_groups_exp.status(), "Failed to configure network group"); return 1; }
    auto network_groups = std::move(network_groups_exp.value());
    auto network_group = network_groups[0];

    auto input_vstream_params = network_group->make_input_vstream_params(false, HAILO_FORMAT_TYPE_UINT8, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!input_vstream_params) { print_error(input_vstream_params.status(), "Failed to make input params"); return 1; }
   
    auto output_vstream_params = network_group->make_output_vstream_params(false, HAILO_FORMAT_TYPE_FLOAT32, HAILO_DEFAULT_VSTREAM_TIMEOUT_MS, HAILO_DEFAULT_VSTREAM_QUEUE_SIZE);
    if (!output_vstream_params) { print_error(output_vstream_params.status(), "Failed to make output params"); return 1; }

    // Split generation to prevent static assertion memory errors
    auto input_vstreams_exp = VStreamsBuilder::create_input_vstreams(*network_group, input_vstream_params.value());
    if (!input_vstreams_exp) { print_error(input_vstreams_exp.status(), "Failed to create input vstreams"); return 1; }
    auto input_vstreams = std::move(input_vstreams_exp.value());

    auto output_vstreams_exp = VStreamsBuilder::create_output_vstreams(*network_group, output_vstream_params.value());
    if (!output_vstreams_exp) { print_error(output_vstreams_exp.status(), "Failed to create output vstreams"); return 1; }
    auto output_vstreams = std::move(output_vstreams_exp.value());

    // 4. Run Inference
    std::cout << "[Running inference...]" << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    if (input_vstreams.size() != 1) {
        std::cerr << "Error: Expected 1 input stream, got " << input_vstreams.size() << std::endl;
        return 1;
    }
   
    hailo_status status = input_vstreams[0].write(MemoryView(resized_image.data, resized_image.total() * resized_image.elemSize()));
    if (status != HAILO_SUCCESS) { print_error(status, "Failed to write input"); return 1; }

    // Read Outputs Dynamically and map safely via shape metadata (NO GUESSING BY SIZE)
    std::map<std::string, std::vector<float>> output_buffers;
    std::vector<const float*> cls_ptrs(3, nullptr);
    std::vector<const float*> reg_ptrs(3, nullptr);
    int num_classes = class_names.size();

    for (auto& ov : output_vstreams) {
        std::string name = ov.name();
        size_t framesize = ov.get_frame_size();
        output_buffers[name] = std::vector<float>(framesize / sizeof(float));
       
        status = ov.read(MemoryView(output_buffers[name].data(), framesize));
        if (status != HAILO_SUCCESS) { print_error(status, "Failed to read output " + name); return 1; }
       
        // This is the critical fix! Reading the shape features to map the tensors properly.
        auto shape = ov.get_info().shape;
        if (shape.features == num_classes) {
            if (shape.height == 80) cls_ptrs[0] = output_buffers[name].data();
            else if (shape.height == 40) cls_ptrs[1] = output_buffers[name].data();
            else if (shape.height == 20) cls_ptrs[2] = output_buffers[name].data();
        } else if (shape.features == 4) {
            if (shape.height == 80) reg_ptrs[0] = output_buffers[name].data();
            else if (shape.height == 40) reg_ptrs[1] = output_buffers[name].data();
            else if (shape.height == 20) reg_ptrs[2] = output_buffers[name].data();
        }
    }
   
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end_time - start_time;
    std::cout << "✓ Inference completed in " << duration.count() << "ms" << std::endl;

    // 5. C++ Post-Processing Verification
    if (!cls_ptrs[0] || !cls_ptrs[1] || !cls_ptrs[2] || !reg_ptrs[0] || !reg_ptrs[1] || !reg_ptrs[2]) {
        std::cerr << "Error: Could not map HEF output tensors based on shape metadata!" << std::endl;
        return 1;
    }

    std::vector<Detection> detections = run_postprocess(
        IntList<8, 16, 32>{},
        IntList<80, 40, 20>{},
        cls_ptrs,
        reg_ptrs,
        conf_threshold,
        num_classes
    );

    std::cout << "✓ Found " << detections.size() << " detections above threshold " << conf_threshold << std::endl;

    // 6. Rescale Detections and Render to Image
    for (const auto& det : detections) {
        // Reverse Letterbox margin mapping
        float x1_f = (det.x1 - lb_info.pad_w) / lb_info.scale;
        float y1_f = (det.y1 - lb_info.pad_h) / lb_info.scale;
        float x2_f = (det.x2 - lb_info.pad_w) / lb_info.scale;
        float y2_f = (det.y2 - lb_info.pad_h) / lb_info.scale;
       
        // Safety Clipping to image bounds
        int x1 = std::max(0, std::min((int)x1_f, orig_image.cols));
        int y1 = std::max(0, std::min((int)y1_f, orig_image.rows));
        int x2 = std::max(0, std::min((int)x2_f, orig_image.cols));
        int y2 = std::max(0, std::min((int)y2_f, orig_image.rows));
       
        cv::rectangle(orig_image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
       
        // Dynamically assign class label text from the parsed vector
        std::string class_name_str = (det.cls_id < class_names.size()) ? class_names[det.cls_id] : "Class_" + std::to_string(det.cls_id);
        std::string label = class_name_str + " " + std::to_string(det.conf).substr(0, 4);
       
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(orig_image, cv::Point(x1, y1 - textSize.height - 5), cv::Point(x1 + textSize.width, y1), cv::Scalar(255, 0, 0), -1);
        cv::putText(orig_image, label, cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
       
        std::cout << "  " << label << " at [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]" << std::endl;
    }

    // 7. Save output payload
    cv::imwrite(output_path, orig_image);
    std::cout << "✓ Output image saved to: " << output_path << std::endl;

    return 0;
}