# ⚡ High-Performance C++ Inference & Tracking Pipeline

This directory contains the highly optimized, C++ runtime environment for deploying custom YOLO26 Nano models on the Hailo-8L accelerator. 

By bypassing Python overhead and executing directly in C++, this pipeline extracts the maximum possible frame rate (FPS) from the Raspberry Pi 5 and the Hailo M.2 HAT+.

---

## 📂 File Manifest & Attributions

This C++ architecture blends robust open-source foundations with custom-engineered, real-time video processing pipelines.

### 1. Adapted Open-Source Components
The following files were originally adapted from the [DanielDubinsky/yolo26_hailo](https://github.com/DanielDubinsky/yolo26_hailo) repository and heavily modified for this project:

* **`detect_image.cpp`**: Originally a static, single-image execution script. **Modifications:** Re-engineered with a custom CLI argument parser to dynamically accept model parameters (like `--labels` and `--model-size`) without recompiling.
* **`include/preprocess.hpp`**: Handles hardware-agnostic image letterboxing and scaling.
* **`include/postprocess.hpp`**: The core Non-Maximum Suppression (NMS) and decoding logic. **Major Modifications:** The original repository hardcoded all tensor shapes and memory offsets to strictly support the 80-class COCO dataset. We completely rewrote the mapping functions (`map_output_tensors` and `run_postprocess`) to calculate buffer sizes dynamically based on an arbitrary `num_classes` variable, allowing seamless support for custom datasets (e.g., 1-class Bottle detection).

### 2. Custom Engineering (Original Work)
The following scripts were engineered from scratch for this repository to support live streams and advanced temporal tracking:

* **`detect_video.cpp`**: A highly optimized live-camera inference loop. It manages OpenCV VideoCapture buffers, real-time FPS telemetry, and dynamic multi-class label rendering over live video.
* **`track_video.cpp`**: An advanced multi-object tracking platform. It integrates external state-of-the-art tracking algorithms, temporal bounding box smoothing (Exponential Moving Average), and association padding inflation to eliminate visual jitter.

---

## 🔗 External Dependencies

Before building the C++ pipeline, ensure your system has the following installed:

1.  **OpenCV (C++)**: `sudo apt install libopencv-dev`
2.  **HailoRT**: Installed via the official `.deb` packages from the Hailo Developer Zone.
3.  **Roboflow Trackers C++ Library**: The `track_video` binary relies on an external, system-level tracking library. 
    * **Repository:** [MatinRafiei/roboflow-trackers-cpp](https://github.com/MatinRafiei/roboflow-trackers-cpp)
    * **Installation:** Please clone that repository, follow its CMake installation instructions to install `libroboflow_trackers.so` to your system path, and then return here.

---

## 🛠️ Build Instructions

We use `CMake` from the **root directory** of this repository to build the C++ executables.

```bash
# 1. Navigate to the ROOT of the repository (not inside the cpp/ folder)
cd /path/to/repo/root

# 2. Create a build directory
mkdir build && cd build

# 3. Generate the build files
cmake ..

# 4. Compile the binaries (using all 4 Raspberry Pi CPU cores for speed)
make -j$(nproc)
```
Once complete, your three executable binaries (detect_image, detect_video, and track_video) will be located inside the build/ directory.

## 🚀 Usage Guide

All binaries feature a dynamic argument parser. You can view all available flags by passing --help.
Single Image Detection

Run inference on a static frame using your custom .hef file:
```bash
./detect_image ../dataset/images/test/sample.jpg \
    --hef ../training/model/bottle.hef \
    --labels bottle \
    --model-size 640 \
    --conf-threshold 0.35
```

### Real-Time Video Detection
Stream directly from a USB Camera (Device `0`) or an IP camera RTSP URL:
```bash
./detect_video \
    --camera-id 0 \
    --hef ../training/model/bottle.hef \
    --labels bottle \
    --input-width 640 \
    --input-height 480
```

### Advanced Object Tracking (BoTSORT)

Run the full tracking pipeline with EMA temporal smoothing and bounding box inflation logic:
```bash
./track_video \
    --camera-id 0 \
    --hef ../training/model/bottle.hef \
    --labels bottle \
    --track-buffer 120 \
    --pad-inflation 40 \
    --alpha 0.7 \
    --debug
```
*(Note: Passing `--debug` will print live jump-distance diagnostics to the terminal if the tracker detects an Identity Switch).*